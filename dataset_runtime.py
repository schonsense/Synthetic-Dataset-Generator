"""Durable dataset ownership for one generation run.

The job manifest is written once. Each newly assigned binding is written in its
own atomic record before it can reach a model request. Main generation-state
snapshots therefore contain a small reference, not repeated copies of every
input passage and dataset cell. Sources are inspected lazily by a worker.

Restored jobs use their saved inputs and values without reopening sources.
Allocating new rows after restoration delegates verified cursor replay to the
binding manager. This resumes whole jobs, not individual conversation turns.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from types import MappingProxyType
from uuid import uuid4

from dataset_bindings import DatasetBindingError, DatasetBindingManager, SourceBinding, TaskBinding
from dataset_settings import normalize_config, source_selector, source_spec, validate_config
from dataset_sources import open_source


class DatasetRuntimeError(DatasetBindingError):
    """A job, source, or durable run checkpoint cannot safely be used."""


_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_DIGEST = "0" * 64
_RUN_CONFIG_KEYS = frozenset((
    "prompts.use_questions_file", "generation.num_turns", "generation.subject_size",
    "generation.context_size", "api.master_duplication_mode", "prompts.opening_mode",
))


def _settings(data):
    normalized = normalize_config(data)
    validate_config(normalized)
    return normalized


def _active_sources(settings):
    return [source for source in settings["sources"]
            if source["enabled"] and source["column_aliases"]]


def has_active_sources(settings):
    """Validate locally and report whether any enabled aliases need a runtime."""
    return bool(_active_sources(_settings(settings)))


def _signature(settings):
    # Presentation changes must not prevent restoring an otherwise identical run.
    sources = [{"source_id": source["source_id"],
                "selector": source_selector(source),
                "column_aliases": source["column_aliases"]}
               for source in _active_sources(settings)]
    return hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _run_config(value):
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - _RUN_CONFIG_KEYS:
        raise DatasetRuntimeError("Run configuration contains unsupported fields.")
    if any(item is not None and type(item) not in (str, bool, int, float) for item in value.values()):
        raise DatasetRuntimeError("Run configuration values must be scalar settings.")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DatasetRuntimeError("Invalid run configuration: {}".format(exc)) from exc
    return MappingProxyType(dict(value))


def _task(value):
    if not isinstance(value, (list, tuple)) or len(value) not in (4, 5):
        raise DatasetRuntimeError("A saved job must be a four- or five-element task tuple.")
    result = tuple(value)
    if (not isinstance(result[0], str) or not result[0]
            or not isinstance(result[1], str) or not result[1]
            or type(result[2]) is not int or result[2] < 0
            or any(not isinstance(item, str) for item in result[3:])):
        raise DatasetRuntimeError("A job contains an invalid ID, source, position, or input text.")
    return result


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _write_atomic(path, content):
    """Replace one local record after flushing its complete UTF-8 JSON bytes."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=str(path.parent),
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
        # Directory fsync is supported on POSIX, but not all Windows/filesystem
        # combinations. File flushing plus atomic replacement remains mandatory.
        if hasattr(os, "O_DIRECTORY"):
            try:
                descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _binding_filename(index, task_id):
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    return "{:08d}-{}.json".format(index, digest)


def _record_digest(record):
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class DatasetRun:
    """Frozen settings, original queued jobs, and one durable binding per job.

    Construct and add jobs on the producer thread; ``seal`` writes the manifest
    before workers start. ``bind_task`` may then run concurrently. A lock covers
    only row allocation and durable assignment, never a model call. Existing
    durable jobs remain usable after source failure or reader closure.
    """

    def __init__(self, settings, state_path, run_config=None, source_factory=None):
        self._settings = _settings(settings)
        self._active = _active_sources(self._settings)
        if not self._active:
            raise DatasetRuntimeError("A dataset run requires at least one enabled mapped source.")
        self.run_id = uuid4().hex
        self.run_config = _run_config(run_config)
        self.settings_signature = _signature(self._settings)
        self._state_path = Path(state_path).resolve()
        self._directory = self._state_path.parent / "dataset_runs" / self.run_id
        self._lock = threading.RLock()
        self._tasks = {}
        self._bindings = {}
        self._manager = None
        self._source_factory = source_factory
        self._sealed = False
        self._storage_created = False
        self._manifest_digest = None
        self._allocation_digest = _EMPTY_DIGEST
        self._closed = False
        self._allocation_failure = None

    @property
    def settings(self):
        return normalize_config(self._settings)

    @property
    def tasks(self):
        with self._lock:
            return tuple(self._tasks.values())

    @property
    def task_ids(self):
        with self._lock:
            return frozenset(self._tasks)

    @property
    def bound_count(self):
        with self._lock:
            return len(self._bindings)

    @property
    def bound_task_ids(self):
        with self._lock:
            return frozenset(self._bindings)

    @property
    def failed(self):
        with self._lock:
            return self._allocation_failure is not None

    def task_for_id(self, task_id):
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise DatasetRuntimeError("Unknown dataset job {!r}.".format(task_id)) from exc

    def add_task(self, task):
        """Assign a unique logical ID, even when two sampled inputs are identical."""
        original = _task(task)
        with self._lock:
            if self._sealed or self._closed:
                raise DatasetRuntimeError("The dataset job catalog is already sealed.")
            task_id = "ds_{}_{:08d}".format(self.run_id, len(self._tasks))
            result = (task_id,) + original[1:]
            self._tasks[task_id] = result
            return result

    def seal(self):
        """Persist the complete original job catalog once; safe to repeat on resume."""
        with self._lock:
            if self._sealed:
                return
            if self._closed:
                raise DatasetRuntimeError("The dataset run is closed.")
            if not self._tasks:
                raise DatasetRuntimeError("The dataset job catalog is empty.")
            manifest = {"version": 1, "run_id": self.run_id, "settings": self._settings,
                        "run_config": dict(self.run_config), "tasks": list(self._tasks.values())}
            content = _json_bytes(manifest)
            try:
                if not self._storage_created:
                    self._directory.mkdir(parents=True, exist_ok=False)
                    self._storage_created = True
                (self._directory / "bindings").mkdir(exist_ok=True)
                _write_atomic(self._directory / "allocation.json", _json_bytes({
                    "version": 1, "count": 0, "last_sha256": _EMPTY_DIGEST,
                }))
                _write_atomic(self._directory / "manifest.json", content)
            except OSError as exc:
                raise DatasetRuntimeError("Could not save the original dataset jobs: {}".format(exc)) from exc
            self._manifest_digest = hashlib.sha256(content).hexdigest()
            self._sealed = True

    def snapshot(self):
        """Return the small reference stored in the main generation-state file."""
        with self._lock:
            if not self._sealed:
                raise DatasetRuntimeError("Dataset jobs must be saved before generation state.")
            return {"version": 1, "run_id": self.run_id,
                    "manifest_sha256": self._manifest_digest}

    def _make_manager(self):
        factory = self._source_factory or open_source
        if self._bindings:
            return DatasetBindingManager.from_bindings(
                tuple(self._bindings.values()), source_factory=factory)

        def selections():
            # Yield one reader at a time so the manager owns construction cleanup.
            for source in self._active:
                yield SourceBinding(factory(source_spec(source)), source["column_aliases"])

        return DatasetBindingManager(self.run_id, selections())

    def _validate_binding(self, binding):
        if binding.run_id != self.run_id or binding.task_id not in self._tasks:
            raise DatasetRuntimeError("A saved binding belongs to a different run or unknown job.")
        if len(binding.sources) != len(self._active):
            raise DatasetRuntimeError("Saved binding sources do not match the run's mapped sources.")
        for source, saved in zip(self._active, binding.sources):
            spec = source_spec(source)
            if (saved.spec.source_id != spec.source_id
                    or saved.spec.kind != spec.kind or saved.spec.location != spec.location
                    or dict(saved.column_aliases) != source["column_aliases"]
                    or (spec.config is not None and saved.spec.config != spec.config)
                    or (spec.split is not None and saved.spec.split != spec.split)):
                raise DatasetRuntimeError("Saved binding source identity or aliases do not match the manifest.")

    def bind_task(self, task):
        """Return values only after their assignment is durable, without resampling."""
        original = _task(task)
        with self._lock:
            task_id = original[0]
            if self._tasks.get(task_id) != original:
                raise DatasetRuntimeError("Dataset job ID or original input payload has changed.")
            if task_id in self._bindings:
                return self._bindings[task_id]
            if not self._sealed:
                raise DatasetRuntimeError("Dataset job inputs have not been checkpointed.")
            if self._closed:
                raise DatasetRuntimeError("The dataset run is closed; new rows cannot be assigned.")
            if self._allocation_failure is not None:
                raise DatasetRuntimeError(self._allocation_failure)
            try:
                if self._manager is None:
                    self._manager = self._make_manager()
                binding = self._manager.bind(task_id)
                self._validate_binding(binding)
                index = len(self._bindings)
                record = {"version": 1, "allocation_index": index,
                          "previous_sha256": self._allocation_digest,
                          "binding": binding.to_dict()}
                record["record_sha256"] = _record_digest(record)
                path = self._directory / "bindings" / _binding_filename(index, task_id)
                content = _json_bytes(record)
                digest = hashlib.sha256(content).hexdigest()
                _write_atomic(path, content)
                _write_atomic(self._directory / "allocation.json", _json_bytes({
                    "version": 1, "count": index + 1, "last_sha256": digest,
                }))
                self._bindings[task_id] = binding
                self._allocation_digest = digest
                return binding
            except Exception as exc:
                self._allocation_failure = "Cannot assign a new dataset row: {}".format(exc)
                raise DatasetRuntimeError(self._allocation_failure) from exc

    @classmethod
    def from_checkpoint(cls, descriptor, state_path, settings, source_factory=None):
        """Restore original jobs and durable values without any dataset source I/O."""
        if (not isinstance(descriptor, dict) or type(descriptor.get("version")) is not int
                or descriptor["version"] != 1
                or not isinstance(descriptor.get("run_id"), str)
                or not _RUN_ID.fullmatch(descriptor["run_id"])
                or not isinstance(descriptor.get("manifest_sha256"), str)
                or not _DIGEST.fullmatch(descriptor["manifest_sha256"])):
            raise DatasetRuntimeError("Unsupported or malformed dataset-run checkpoint reference.")
        state_path = Path(state_path).resolve()
        directory = state_path.parent / "dataset_runs" / descriptor["run_id"]
        try:
            content = (directory / "manifest.json").read_bytes()
            if hashlib.sha256(content).hexdigest() != descriptor["manifest_sha256"]:
                raise DatasetRuntimeError("The saved original-job manifest has changed or is incomplete.")
            manifest = json.loads(content.decode("utf-8"))
            if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
                    or manifest["version"] != 1 or manifest.get("run_id") != descriptor["run_id"]
                    or not isinstance(manifest.get("tasks"), list) or not manifest["tasks"]):
                raise DatasetRuntimeError("Malformed dataset job manifest.")
            restored = cls(manifest["settings"], state_path,
                           run_config=manifest["run_config"], source_factory=source_factory)
            if restored.settings_signature != _signature(_settings(settings)):
                raise DatasetRuntimeError("Dataset source settings or aliases differ from the saved run.")
            restored.run_id = descriptor["run_id"]
            restored._directory = directory
            restored._manifest_digest = descriptor["manifest_sha256"]
            restored._storage_created = restored._sealed = True
            for ordinal, value in enumerate(manifest["tasks"]):
                task = _task(value)
                if task[0] != "ds_{}_{:08d}".format(restored.run_id, ordinal):
                    raise DatasetRuntimeError("Saved dataset jobs have duplicate or inconsistent IDs.")
                restored._tasks[task[0]] = task
            binding_directory = directory / "bindings"
            if not binding_directory.is_dir():
                raise DatasetRuntimeError("The dataset binding checkpoint directory is missing.")
            frontier = json.loads((directory / "allocation.json").read_text(encoding="utf-8"))
            if (not isinstance(frontier, dict) or type(frontier.get("version")) is not int
                    or frontier["version"] != 1 or type(frontier.get("count")) is not int
                    or frontier["count"] < 0 or not isinstance(frontier.get("last_sha256"), str)
                    or not _DIGEST.fullmatch(frontier["last_sha256"])):
                raise DatasetRuntimeError("The dataset allocation checkpoint is invalid.")
            paths = sorted(binding_directory.glob("*.json"))
            if not frontier["count"] <= len(paths) <= frontier["count"] + 1:
                raise DatasetRuntimeError("A dataset binding checkpoint is missing or has an invalid allocation order.")
            if frontier["count"] == 0 and frontier["last_sha256"] != _EMPTY_DIGEST:
                raise DatasetRuntimeError("The empty dataset allocation checkpoint is invalid.")
            for index, path in enumerate(paths):
                binding_content = path.read_bytes()
                record = json.loads(binding_content.decode("utf-8"))
                if (not isinstance(record, dict) or type(record.get("version")) is not int
                        or record["version"] != 1 or type(record.get("allocation_index")) is not int
                        or record["allocation_index"] != index
                        or record.get("previous_sha256") != restored._allocation_digest):
                    raise DatasetRuntimeError("Saved dataset allocation order is incomplete or invalid.")
                if record.get("record_sha256") != _record_digest(record):
                    raise DatasetRuntimeError("A saved dataset binding record is corrupt.")
                restored._allocation_digest = hashlib.sha256(binding_content).hexdigest()
                if index + 1 == frontier["count"] and restored._allocation_digest != frontier["last_sha256"]:
                    raise DatasetRuntimeError("A saved dataset binding changed after allocation.")
                binding = TaskBinding.from_dict(record["binding"])
                if path.name != _binding_filename(index, binding.task_id):
                    raise DatasetRuntimeError("A saved dataset binding filename does not match its job.")
                restored._validate_binding(binding)
                if binding.task_id in restored._bindings:
                    raise DatasetRuntimeError("A saved dataset job has more than one row assignment.")
                restored._bindings[binding.task_id] = binding
            if restored._bindings:
                # Validate cross-record source consistency before adopting any
                # crash-orphan record into the durable allocation frontier.
                restored._manager = restored._make_manager()
            if len(paths) > frontier["count"]:
                # A crash can leave one complete record just before its frontier
                # update. It never reached a request, but retaining that assignment
                # is safe. Commit its frontier before permitting another allocation.
                _write_atomic(directory / "allocation.json", _json_bytes({
                    "version": 1, "count": len(paths),
                    "last_sha256": restored._allocation_digest,
                }))
            return restored
        except Exception as exc:
            if isinstance(exc, DatasetRuntimeError):
                raise
            raise DatasetRuntimeError("Could not restore dataset jobs and values: {}".format(exc)) from exc

    def close(self):
        """Release readers after workers finish; retain saved values and records."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._manager is not None:
                self._manager.close()

    def discard_checkpoint(self):
        """Remove only this run's records after an explicit Stop & Clear action."""
        with self._lock:
            self.close()
            expected = self._state_path.parent / "dataset_runs" / self.run_id
            if self._directory != expected or not _RUN_ID.fullmatch(self.run_id):
                raise DatasetRuntimeError("Refusing to remove an unexpected dataset checkpoint path.")
            if self._storage_created and self._directory.exists():
                shutil.rmtree(str(self._directory))
            self._storage_created = False
