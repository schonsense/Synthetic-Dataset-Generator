"""Durable run tests: python -B -m unittest test_dataset_runtime -v.

Real temporary CSV/JSONL sources and checkpoint files are used. Filesystem faults
are injected at atomic replacement boundaries. No GUI, network, application
configuration, or model request is involved; all data belongs to the test's
temporary directory.
"""

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import dataset_runtime
from dataset_runtime import DatasetRun, DatasetRuntimeError, has_active_sources
from dataset_sources import DatasetSource, open_source


class CountingSource(DatasetSource):
    def __init__(self, source):
        super().__init__(source.spec)
        self.source = source
        self.schema_calls = 0
        self.row_ids = []
        self.close_calls = 0

    def schema(self):
        self.schema_calls += 1
        return self.source.schema()

    def iter_rows(self):
        iterator = self.source.iter_rows()
        try:
            for row in iterator:
                self.row_ids.append(row.row_id)
                yield row
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def close(self):
        self.close_calls += 1
        self.source.close()


class SourceFactory:
    def __init__(self):
        self.created = []

    def __call__(self, spec):
        source = CountingSource(open_source(spec))
        self.created.append(source)
        return source


def source_setting(source_id, kind, path, columns, aliases):
    selector = {"kind": kind, "location": str(path), "config": None, "split": None, "revision": None}
    return {
        "source_id": source_id, "name": source_id.title(), "enabled": True,
        **selector, "columns": list(columns), "column_aliases": dict(aliases),
        "visible_columns": None, "schema_selector": dict(selector),
        "schema_fingerprint": "previously-imported-schema-" + source_id,
    }


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def refresh_record_checksum(record):
    """Independently encode the documented self-checksum, to test deeper checks."""
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(content).hexdigest()


class DatasetRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.fixture_number = 0

    def fixture(self, count=24, note=None):
        self.fixture_number += 1
        directory = self.directory / ("fixture-{}".format(self.fixture_number))
        directory.mkdir()
        records_path = directory / "records.csv"
        policies_path = directory / "policies.jsonl"
        records = [{
            "record_id": "record-{:03d}".format(index), "label": "Label {:03d}".format(index),
            "note": note if note is not None else 'literal {{ds:record_key}} {context}\n"é" C:\\records\\' + str(index),
        } for index in range(count)]
        policies = [{"policy": "policy-{:03d}".format(index), "limits": [index, None]} for index in range(count)]
        with records_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("record_id", "label", "note"))
            writer.writeheader()
            writer.writerows(records)
        policies_path.write_text("".join(json.dumps(value) + "\n" for value in policies), encoding="utf-8")
        settings = {"sources": [
            source_setting("records", "csv", records_path, ("record_id", "label", "note"),
                           {"record_id": "record_key", "label": "record_label", "note": "record_note"}),
            source_setting("policies", "jsonl", policies_path, ("policy", "limits"),
                           {"policy": "policy", "limits": "limits"}),
        ]}
        return {"settings": settings, "state_path": directory / "output" / "generation_state.json",
                "records": records, "policies": policies, "paths": (records_path, policies_path)}

    def new_run(self, fixture, factory=None, run_config=None):
        run = DatasetRun(fixture["settings"], fixture["state_path"], run_config, source_factory=factory)
        self.addCleanup(run.close)
        return run

    def add_jobs(self, run, count, subject="original subject", context="original context"):
        return [run.add_task(("sample-{}".format(index), "input.txt", index, subject, context))
                for index in range(count)]

    @staticmethod
    def checkpoint_directory(fixture, run):
        return fixture["state_path"].parent / "dataset_runs" / run.run_id

    def restore(self, fixture, descriptor, settings=None, factory=None):
        run = DatasetRun.from_checkpoint(descriptor, fixture["state_path"],
                                         fixture["settings"] if settings is None else settings,
                                         source_factory=factory)
        self.addCleanup(run.close)
        return run

    def assert_rows_allocated(self, factory, count):
        self.assertEqual([source.spec.source_id for source in factory.created], ["records", "policies"])
        for source in factory.created:
            self.assertEqual(source.row_ids, [str(index) for index in range(count)])

    def test_sources_are_lazy_and_disabled_or_unmapped_drafts_are_ignored(self):
        fixture = self.fixture()
        disabled = {"source_id": "disabled", "enabled": False, "location": "missing.csv",
                    "column_aliases": {"absent": "invalid-alias"}}
        unmapped = {"source_id": "draft", "enabled": True, "location": "", "column_aliases": {}}
        fixture["settings"]["sources"].extend([disabled, unmapped])
        self.assertFalse(has_active_sources(None))
        self.assertFalse(has_active_sources({"sources": [disabled, unmapped]}))
        self.assertTrue(has_active_sources(fixture["settings"]))
        factory = SourceFactory()
        run = self.new_run(fixture, factory)
        self.assertFalse(self.checkpoint_directory(fixture, run).exists())
        job = self.add_jobs(run, 1)[0]
        run.seal()
        run.snapshot()
        self.assertEqual(factory.created, [], "Schema discovery must wait until a worker first binds.")
        binding = run.bind_task(job)
        self.assertEqual(binding.values["record_key"], "record-000")
        self.assert_rows_allocated(factory, 1)

    def test_identical_original_inputs_get_unique_stable_logical_ids(self):
        fixture = self.fixture()
        run = self.new_run(fixture)
        original = ("same-chunk", "input.txt", 0, 'quoted "subject"\n', "context \\ {{ds:literal}}")
        first, second = run.add_task(original), run.add_task(original)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1:], original[1:])
        self.assertEqual(second[1:], original[1:])
        self.assertEqual(run.task_ids, frozenset((first[0], second[0])))
        self.assertEqual(run.task_for_id(first[0]), first)
        run.seal()
        restored = self.restore(fixture, run.snapshot(), factory=Mock(side_effect=AssertionError("Source I/O")))
        self.assertEqual(restored.tasks, (first, second))
        with self.assertRaises(DatasetRuntimeError):
            run.add_task(original)

    def test_hidden_aliases_and_settings_are_frozen_but_gui_presentation_does_not_block_restore(self):
        fixture = self.fixture()
        fixture["settings"]["sources"][0]["visible_columns"] = ["record_id"]
        original_settings = copy.deepcopy(fixture["settings"])
        run_config = {"generation.num_turns": 2}
        run = self.new_run(fixture, run_config=run_config)
        job = self.add_jobs(run, 1)[0]
        run.seal()
        fixture["settings"]["sources"][0]["column_aliases"]["label"] = "edited_alias"
        fixture["settings"]["sources"][0]["location"] = "missing.csv"
        detached = run.settings
        detached["sources"][1]["enabled"] = False
        run_config["generation.num_turns"] = 99
        binding = run.bind_task(job)
        self.assertEqual(binding.values["record_label"], "Label 000")
        self.assertNotIn("edited_alias", binding.values)
        self.assertEqual(dict(run.run_config), {"generation.num_turns": 2})

        presentation = copy.deepcopy(original_settings)
        presentation["sources"][0].update(name="Renamed in GUI", visible_columns=[],
                                           columns=["note", "label", "record_id", "extra_cached_name"],
                                           schema_fingerprint="another-metadata-import")
        presentation["sources"].append({"source_id": "later-draft", "enabled": False, "location": ""})
        factory = Mock(side_effect=AssertionError("Presentation-only changes caused source I/O"))
        restored = self.restore(fixture, run.snapshot(), settings=presentation, factory=factory)
        self.assertEqual(restored.bind_task(job), binding)
        factory.assert_not_called()
        changed_mapping = copy.deepcopy(original_settings)
        changed_mapping["sources"][0]["column_aliases"]["label"] = "different_alias"
        with self.assertRaisesRegex(DatasetRuntimeError, "settings or aliases differ"):
            self.restore(fixture, run.snapshot(), settings=changed_mapping)

    def test_twenty_concurrent_jobs_have_one_durable_binding_and_correlated_values_each(self):
        fixture = self.fixture()
        factory = SourceFactory()
        run = self.new_run(fixture, factory)
        jobs = self.add_jobs(run, 20)
        run.seal()
        barrier = threading.Barrier(20)

        def bind(job):
            barrier.wait(timeout=10)
            return run.bind_task(job)

        with ThreadPoolExecutor(max_workers=20) as pool:
            bindings = list(pool.map(bind, jobs))
        self.assertEqual(run.bound_count, 20)
        self.assertEqual({binding.values["record_key"] for binding in bindings},
                         {"record-{:03d}".format(index) for index in range(20)})
        for job, binding in zip(jobs, bindings):
            self.assertEqual(binding.task_id, job[0])
            index = int(binding.values["record_key"].split("-")[1])
            self.assertEqual(binding.values["record_label"], "Label {:03d}".format(index))
            self.assertEqual(binding.values["policy"], "policy-{:03d}".format(index))
            self.assertEqual(binding.values["limits"], "[{},null]".format(index))
            self.assertIs(run.bind_task(job), binding)
        self.assert_rows_allocated(factory, 20)
        restored = self.restore(fixture, run.snapshot(), factory=Mock(side_effect=AssertionError("Restore read sources")))
        self.assertEqual(restored.bound_count, 20)
        for job, binding in zip(jobs, bindings):
            self.assertEqual(restored.bind_task(job), binding)

    def test_simultaneous_calls_for_one_job_commit_only_one_assignment(self):
        fixture = self.fixture()
        factory = SourceFactory()
        run = self.new_run(fixture, factory)
        job = self.add_jobs(run, 1)[0]
        run.seal()
        barrier = threading.Barrier(20)

        def bind(_):
            barrier.wait(timeout=10)
            return run.bind_task(job)

        with ThreadPoolExecutor(max_workers=20) as pool:
            bindings = list(pool.map(bind, range(20)))
        self.assertTrue(all(binding is bindings[0] for binding in bindings))
        self.assertEqual(run.bound_count, 1)
        directory = self.checkpoint_directory(fixture, run)
        self.assertEqual(len(list((directory / "bindings").glob("*.json"))), 1)
        self.assertEqual(json.loads((directory / "allocation.json").read_text())["count"], 1)
        self.assert_rows_allocated(factory, 1)

    def test_restored_bound_jobs_work_without_sources_and_new_jobs_resume_next_row(self):
        fixture = self.fixture()
        run = self.new_run(fixture)
        jobs = self.add_jobs(run, 3)
        run.seal()
        first, second = run.bind_task(jobs[1]), run.bind_task(jobs[0])
        descriptor = run.snapshot()
        run.close()
        # Queue order differs from allocation order; durable ordinals must win.
        factory = SourceFactory()
        restored = self.restore(fixture, descriptor, factory=factory)
        self.assertEqual(restored.bind_task(jobs[1]), first)
        self.assertEqual(restored.bind_task(jobs[0]), second)
        self.assertEqual(factory.created, [])
        third = restored.bind_task(jobs[2])
        self.assertEqual(third.values["record_key"], "record-002")
        self.assert_rows_allocated(factory, 3)
        restored.close()
        for path in fixture["paths"]:
            path.unlink()
        offline_factory = Mock(side_effect=AssertionError("A bound restore reopened missing sources"))
        offline = self.restore(fixture, descriptor, factory=offline_factory)
        self.assertEqual(offline.bind_task(jobs[2]), third)
        self.assertEqual(offline.tasks, tuple(jobs))
        offline_factory.assert_not_called()

    def test_missing_or_changed_source_blocks_only_new_allocations(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                fixture = self.fixture()
                run = self.new_run(fixture)
                jobs = self.add_jobs(run, 2)
                run.seal()
                original = run.bind_task(jobs[0])
                descriptor = run.snapshot()
                run.close()
                if missing:
                    fixture["paths"][0].unlink()
                else:
                    fixture["paths"][0].write_text("record_id,label,note\nchanged,changed,changed\n", encoding="utf-8")
                factory = SourceFactory()
                restored = self.restore(fixture, descriptor, factory=factory)
                self.assertEqual(restored.bind_task(jobs[0]), original)
                for _ in range(2):
                    with self.assertRaisesRegex(DatasetRuntimeError, "Cannot assign a new dataset row"):
                        restored.bind_task(jobs[1])
                self.assertTrue(restored.failed)
                self.assertEqual(restored.bound_count, 1)
                self.assertEqual(restored.bind_task(jobs[0]), original)
                self.assertEqual(len(factory.created), 1, "A latched error reopened its failed source.")

    def test_integrity_errors_reject_restore_before_source_access(self):
        for corruption in ("missing-last", "chain", "manifest", "count"):
            with self.subTest(corruption=corruption):
                fixture = self.fixture()
                run = self.new_run(fixture)
                jobs = self.add_jobs(run, 2)
                run.seal()
                for job in jobs:
                    run.bind_task(job)
                descriptor = run.snapshot()
                run.close()
                directory = self.checkpoint_directory(fixture, run)
                paths = sorted((directory / "bindings").glob("*.json"))
                if corruption == "missing-last":
                    paths[-1].unlink()
                elif corruption == "chain":
                    record = json.loads(paths[0].read_text(encoding="utf-8"))
                    record["binding"]["values"]["record_note"] = "changed valid JSON"
                    refresh_record_checksum(record)  # Reach the cross-record hash-chain check.
                    paths[0].write_bytes(encode_json(record))
                elif corruption == "manifest":
                    path = directory / "manifest.json"
                    path.write_bytes(path.read_bytes()[:-3])
                else:
                    frontier = json.loads((directory / "allocation.json").read_text())
                    frontier["count"] = True
                    (directory / "allocation.json").write_bytes(encode_json(frontier))
                factory = Mock(side_effect=AssertionError("Invalid checkpoint accessed a source"))
                with self.assertRaises(DatasetRuntimeError):
                    self.restore(fixture, descriptor, factory=factory)
                factory.assert_not_called()

    def test_first_record_durability_failure_exposes_no_binding_and_latches_allocation(self):
        fixture = self.fixture()
        factory = SourceFactory()
        run = self.new_run(fixture, factory)
        jobs = self.add_jobs(run, 2)
        run.seal()
        real_replace = os.replace

        def fail_binding_replace(source, destination):
            if Path(destination).parent.name == "bindings":
                raise OSError("simulated binding replacement failure")
            return real_replace(source, destination)

        with patch.object(dataset_runtime.os, "replace", side_effect=fail_binding_replace):
            with self.assertRaisesRegex(DatasetRuntimeError, "binding replacement failure"):
                run.bind_task(jobs[0])
        self.assertEqual(run.bound_count, 0)
        self.assertTrue(run.failed)
        for job in jobs:
            with self.assertRaisesRegex(DatasetRuntimeError, "binding replacement failure"):
                run.bind_task(job)
        directory = self.checkpoint_directory(fixture, run)
        self.assertEqual(list((directory / "bindings").iterdir()), [])
        self.assertEqual(json.loads((directory / "allocation.json").read_text())["count"], 0)
        self.assert_rows_allocated(factory, 1)

    def make_orphan(self, fixture, run, job):
        real_replace = os.replace

        def fail_frontier_replace(source, destination):
            if Path(destination).name == "allocation.json":
                raise OSError("simulated crash before frontier replacement")
            return real_replace(source, destination)

        with patch.object(dataset_runtime.os, "replace", side_effect=fail_frontier_replace):
            with self.assertRaisesRegex(DatasetRuntimeError, "before frontier replacement"):
                run.bind_task(job)
        run.close()
        directory = self.checkpoint_directory(fixture, run)
        return directory, sorted((directory / "bindings").glob("*.json"))[-1]

    def test_one_complete_orphan_is_adopted_without_resampling(self):
        fixture = self.fixture()
        run = self.new_run(fixture)
        job = self.add_jobs(run, 1)[0]
        run.seal()
        descriptor = run.snapshot()
        directory, orphan = self.make_orphan(fixture, run, job)
        self.assertEqual(run.bound_count, 0)
        self.assertEqual(json.loads((directory / "allocation.json").read_text())["count"], 0)
        self.assertTrue(orphan.is_file())
        factory = Mock(side_effect=AssertionError("Orphan adoption resampled its source"))
        restored = self.restore(fixture, descriptor, factory=factory)
        self.assertEqual(restored.bound_count, 1)
        self.assertEqual(restored.bind_task(job).values["record_key"], "record-000")
        self.assertEqual(json.loads((directory / "allocation.json").read_text())["count"], 1)
        factory.assert_not_called()

    def test_corrupt_or_inconsistent_orphans_are_not_adopted(self):
        for corruption in ("content", "source-consistency", "two-orphans"):
            with self.subTest(corruption=corruption):
                fixture = self.fixture()
                run = self.new_run(fixture)
                jobs = self.add_jobs(run, 2)
                run.seal()
                if corruption == "source-consistency":
                    run.bind_task(jobs[0])
                    orphan_job = jobs[1]
                else:
                    orphan_job = jobs[0]
                descriptor = run.snapshot()
                directory, orphan = self.make_orphan(fixture, run, orphan_job)
                frontier_before = (directory / "allocation.json").read_bytes()
                record = json.loads(orphan.read_text(encoding="utf-8"))
                if corruption == "content":
                    record["binding"]["values"]["record_note"] = "corrupted but valid JSON"
                    orphan.write_bytes(encode_json(record))  # Deliberately retain the original self-checksum.
                elif corruption == "source-consistency":
                    record["binding"]["sources"][0]["fingerprint"] = "different-source-version"
                    refresh_record_checksum(record)  # Reach consistency validation, past record integrity.
                    orphan.write_bytes(encode_json(record))
                else:
                    (directory / "bindings" / "additional-orphan.json").write_bytes(orphan.read_bytes())
                factory = Mock(side_effect=AssertionError("Invalid orphan accessed source data"))
                with self.assertRaises(DatasetRuntimeError):
                    self.restore(fixture, descriptor, factory=factory)
                self.assertEqual((directory / "allocation.json").read_bytes(), frontier_before)
                factory.assert_not_called()

    def test_main_snapshot_is_small_while_original_jobs_and_cells_are_preserved(self):
        payload = "LARGE-DATASET-CONTENT\n" + "é" * 100000
        original_input = "LARGE-ORIGINAL-INPUT\n" + "input " * 20000
        fixture = self.fixture(count=1, note=payload)
        run = self.new_run(fixture)
        job = self.add_jobs(run, 1, subject=original_input)[0]
        run.seal()
        binding = run.bind_task(job)
        main_state = encode_json({"dataset_run": run.snapshot()})
        self.assertLess(len(main_state), 300)
        self.assertNotIn(b"LARGE-DATASET-CONTENT", main_state)
        self.assertNotIn(b"LARGE-ORIGINAL-INPUT", main_state)
        restored = self.restore(fixture, run.snapshot(), factory=Mock(side_effect=AssertionError("Source I/O")))
        self.assertEqual(restored.task_for_id(job[0])[3], original_input)
        self.assertEqual(restored.bind_task(job).values["record_note"], payload)
        self.assertEqual(restored.bind_task(job), binding)

    def test_close_preserves_records_and_discard_removes_only_its_own_run(self):
        fixture = self.fixture()
        first = self.new_run(fixture)
        second = self.new_run(fixture)
        first_job, second_job = self.add_jobs(first, 1)[0], self.add_jobs(second, 1)[0]
        first.seal()
        second.seal()
        first_binding = first.bind_task(first_job)
        second.bind_task(second_job)
        first_directory = self.checkpoint_directory(fixture, first)
        second_directory = self.checkpoint_directory(fixture, second)
        foreign = first_directory.parent / "unrelated-folder"
        foreign.mkdir()
        (foreign / "keep.txt").write_text("keep", encoding="utf-8")
        fixture["state_path"].write_text("separate main state", encoding="utf-8")
        first.close()
        self.assertTrue(first_directory.exists())
        self.assertIs(first.bind_task(first_job), first_binding)
        first.discard_checkpoint()
        self.assertFalse(first_directory.exists())
        self.assertTrue((second_directory / "manifest.json").is_file())
        self.assertEqual((foreign / "keep.txt").read_text(), "keep")
        self.assertEqual(fixture["state_path"].read_text(), "separate main state")

    def test_invalid_or_changed_job_payloads_are_rejected_without_consuming_a_row(self):
        fixture = self.fixture()
        factory = SourceFactory()
        run = self.new_run(fixture, factory)
        job = self.add_jobs(run, 1)[0]
        run.seal()
        invalid = [None, (), (job[0], job[1], True, job[3], job[4]),
                   (job[0], job[1], job[2], "changed subject", job[4]),
                   ("unknown-task",) + job[1:]]
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(DatasetRuntimeError):
                run.bind_task(candidate)
        self.assertEqual(factory.created, [])
        self.assertEqual(run.bound_count, 0)
        self.assertFalse(run.failed)
        self.assertEqual(run.bind_task(job).values["record_key"], "record-000")
        self.assert_rows_allocated(factory, 1)


if __name__ == "__main__":
    unittest.main()
