# Dataset bindings: foundation and validation

This is an isolated foundation on `feature/dataset-bindings`, with Step 2 reader,
dependency, and orchestration validation completed. The application does not
import these modules yet. There is no new GUI, generation behavior, configuration
key, or automatic application resume at this checkpoint.

## Placeholder syntax

Use `{{ds:alias}}`, for example `{{ds:nvd_f_name}}`. The future GUI will store the
alias alone (`nvd_f_name`), independently of the exact source column name.
Aliases are case-sensitive and match `[A-Za-z_][A-Za-z0-9_]*`. They must be unique
across the sources used in a run.

The `ds:` namespace separates dataset variables from the existing `{context}`,
`{subject}`, and other single-brace variables. A renderer receives both kinds
of values and scans the original template once. Inserted text is never scanned
again, even if it contains either placeholder syntax. Unknown dataset aliases
and malformed reserved placeholders are errors. Unrelated braces, JSON, pipes,
and tags are preserved. There is no escape syntax for a literal dataset token
in an authored template at this stage; such tokens in inserted values are literal.

Rendering must happen at the point where a stage has both its native values and
the job's dataset binding. Do not run old `.replace()` passes afterward, and do
not treat generated text or conversation history as templates. The eventual
integration must preserve existing behavior on the no-datasets path.

## Source interface

`dataset_sources.py` exposes immutable `SourceSpec`, `DatasetSchema`, and
`DatasetRow` records, plus `open_source(spec)`.

| Kind | Input | Discovery / reading |
| --- | --- | --- |
| `csv`, `tsv` | UTF-8 file, with optional BOM | Read exact header names; stream rows |
| `jsonl` | One JSON object per nonblank line | Scan keys across all rows; stream rows during binding |
| `json` | A JSON array of objects | Load the array in memory; prefer JSONL for large inputs |
| `parquet` | Local Parquet file | Schema metadata; batched rows; optional PyArrow |
| `huggingface` | `owner/dataset` plus config/split/revision | Pin the repository to a commit; schema metadata and streamed rows; optional Datasets backend |
| `hf_disk` | Directory created by `Dataset.save_to_disk()` | Select one saved DatasetDict split or a single Dataset; optional Datasets backend |

Creating a source opens nothing. `source.schema()` returns column names and
metadata, never row values. JSON key discovery must read the file, but does not
retain values in the returned schema. It should run off the GUI thread later.
The file readers handle exact top-level column names; dotted names are literal
columns, not nested field paths. Nested JSON cells serialize as JSON when bound.

Paths, quotes, backslashes, and exact column names are preserved. This code does
not use the application's destructive input sanitizer. Duplicate headers/keys,
wrong row shapes, missing mapped fields, and unsupported cell types are explicit
errors. Sparse JSON records can expose a column in their schema but still fail
if a selected record lacks that mapped column.
Opening a CSV/TSV source raises Python's process-wide CSV field-size limit to
accommodate long text cells; importing this module alone does not change it.

Local fingerprints use file identity, size, and modification time (saved
directories use a file list with sizes/times). They detect ordinary edits, not
all possible content changes. Already-bound values remain valid after a source
changes; attempting to read further rows from a changed local source fails.
Remote schemas carry the resolved commit/config/split as well as a fingerprint.
Remote files outside the repository may have their own mutability limitations.

Hub sources require supported data files and declared feature metadata at the
chosen revision. There is no fallback to a different revision or a preview row
if metadata is missing. Multiple configurations/splits require an explicit choice.
These are text/tabular readers; no automatic image/audio-to-text conversion is
provided. Streaming can still download a full file for some underlying formats.

## Per-job ownership

```python
from dataset_sources import SourceSpec, open_source
from dataset_bindings import DatasetBindingManager, SourceBinding, TaskBinding
from dataset_templates import render_template

people = open_source(SourceSpec("people", "csv", "people.csv"))
content = open_source(SourceSpec("content", "jsonl", "content.jsonl"))

with DatasetBindingManager("unique-run-id", [
    SourceBinding(people, {"first_name": "first", "last_name": "last"}),
    SourceBinding(content, {"body": "body"}),
]) as bindings:
    job = bindings.bind("unique-job-id")
    prompt = render_template(
        "{{ds:first}} {{ds:last}}: {{ds:body}}\n{context}",
        job.values,
        {"context": "The stage's existing context."},
    )
    assert bindings.bind("unique-job-id") is job
    restored = TaskBinding.from_dict(job.to_dict())
    assert restored.values == job.values
```

Supply any number of sources, with globally unique source IDs and aliases.
Only mapped/enabled sources belong in the manager. Each source contributes its
next row to each new job; every mapped column from that source comes from that
one row. Sources have separate cursors. This first policy pairs their sequential
positions; it does not perform joins, Cartesian products, random sampling, or
silent cycling. Exhaustion fails clearly.

Allocation is coordinated across workers, while rendering and model requests
take place after the allocation lock is released. Candidate rows are staged
until all sources succeed. Repeated `bind(job_id)` calls return the original
immutable values; they do not allocate new rows. Source readers and manager
instances belong to one run and must not be shared between runs. Job IDs must
uniquely identify logical generation items, regardless of worker or API slot.

Strings are preserved verbatim. Top-level null becomes an empty string. Other
JSON-compatible cells become compact JSON with sorted object keys and Unicode
preserved. Nulls inside JSON stay `null`; booleans become `true`/`false`.
Non-finite numbers and Python-only objects fail rather than using their repr.

`TaskBinding.to_dict()` stores actual mapped strings, mappings, row IDs, and source
provenance. `from_dict()` restores that binding without reading sources. This
does not yet restore the application's pending jobs, source cursor positions,
or conversations. The application's existing saved-state mechanism still needs
explicit integration. `manager.close()` releases readers while keeping existing
bindings readable; new allocations are then prohibited.

## Optional dependencies and compatibility boundary

All foundation modules and CSV/TSV/JSON/JSONL readers use Python's standard
library and remain compatible with the application's Python 3.8 language target.
Optional packages are imported only when their reader is used. The normal
`requirements.txt` installation does not include Hugging Face or PyArrow.

The optional installation includes both the application and dataset dependencies:

```sh
python -m pip install -r requirements-datasets.txt
```

This requires Python 3.9 or later and pins `datasets==4.4.2`. Its dependencies
include PyArrow, so it covers local Parquet and saved-HF folders too. Datasets
4.x does not execute dataset scripts. Later releases can raise the Python minimum,
which is why the tested optional backend is pinned explicitly here.

The application's Requests requirement is now `requests>=2.32.4,<3`, resolving
the conflict between its former `2.31.0` pin and Datasets' `>=2.32.2` requirement.
Pip can select a release compatible with the interpreter: Requests 2.32.4 supports
Python 3.8; newer releases raise that minimum. This preserves the normal Python
3.8 installation path without silently omitting the optional backend if someone
tries to install it on an unsupported interpreter. The combined install and
existing application regressions were tested on Python 3.12, not every interpreter.

Authentication uses Hugging Face's normal environment or saved-login mechanism;
no access token is included in source settings or binding snapshots. No login,
Hub request, or model API call occurs just by importing these modules.

Primary API references:

- [Datasets 4.0 loading and streaming API](https://huggingface.co/docs/datasets/v4.0.0/package_reference/loading_methods)
- [Datasets 4.4.2 declared dependencies](https://github.com/huggingface/datasets/blob/4.4.2/setup.py)
- [Hugging Face repository metadata API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.dataset_info)
- [Requests release and Python support history](https://requests.readthedocs.io/en/latest/community/updates/)

## Verification and next boundary

Run the dependency-free foundation checks with:

```sh
python -B -m unittest test_dataset_sources test_dataset_bindings test_dataset_templates test_dataset_pipeline
```

After installing the optional requirements, also run the real local backend tests:

```sh
python -B -m unittest test_dataset_backends
```

Those tests use actual Parquet files and `Dataset.save_to_disk()` fixtures. They
skip explicitly when optional packages are absent; a skipped suite does not
count as backend validation. All automated suites avoid network and model calls.

A live Hub check is deliberately opt-in and displays names/provenance, not cells:

```sh
python -B tools/check_dataset_hub.py --repository cornell-movie-review-data/rotten_tomatoes --config default --split train --column text --column label --rows 2
```

### Step 2 evidence (2026-10-28)

The isolated Linux/Python 3.12.14 environment resolved Requests 2.34.2, Datasets
4.4.2, PyArrow 26.0.0, and huggingface-hub 1.33.0. `pip check` reported no broken
requirements. The combined requirements installed in a clean temporary environment.

- 48 foundation tests cover exact columns, correlated rows, immutable bindings,
  source exhaustion/edits, snapshot serialization, concurrency, and literal text.
- 13 tests used real Parquet and saved-HF fixtures, including a Parquet batch
  boundary, saved NumPy formatting, DatasetDict splits, and reader closure.
- 7 orchestration simulation tests exercised six request stages, transient
  failures, whole-job handoff, caching, and source-free snapshot replay.
- Twenty concurrent jobs produced 120 simulated requests while consuming exactly
  twenty rows per source. Twenty concurrent callers for one job consumed one row
  per source; binding reuse is distinct from deduplicating job scheduling.
- Request retries reused identical serialized bodies. A failed job retained its
  binding when another live worker processed a different job before retrying it.
- Saved binding plus original job payload reproduced the original requests after
  source files were removed. This is replay of a job, not continuation recovery.
- The live Hub check read two rows successfully from the public test repository
  at revision `aa13bc287fa6fcab6daf52f0dfb9994269ffea28`, retaining its `default`
  configuration, `train` split, and `text`/`label` columns through binding/replay.
- The existing API-profile, persona/arc, and opening-mode regression scripts pass
  with the combined dependency environment. The opening-mode script contained a
  stale legacy-output fixture in the upstream baseline: it expected no metadata
  while the writer already defaulted to including it. Its one legacy-format call
  now explicitly sets `include_metadata=False`; production output is unchanged.

The orchestration harness is explicitly a simulation using the actual source,
binding, and rendering modules. It does not test their integration into
`generation.worker`, application recovery, or the production cache, because
those hooks are not wired yet. Existing regression scripts exercise the current
worker independently. No paid model calls, GUI interaction, Windows execution,
repository push, or PR was part of this validation checkpoint.

Next is the isolated Datasets panel with background schema discovery. Application
configuration, worker/rendering hooks, unique queued-job IDs, and persisted
bindings plus original job payloads remain subsequent integration work.
