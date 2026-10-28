# Dataset bindings: foundation milestone

This is an isolated foundation on `feature/dataset-bindings`. The application
does not import these modules yet. There is no new GUI, generation behavior,
configuration key, or automatic application resume in this milestone.

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
Optional packages are imported only when their reader is used. They have not
been added to the application's requirements in this milestone.

The Hugging Face adapter targets Datasets 4.x, which does not execute dataset
scripts. An exact candidate for dependency validation is `datasets==4.4.2`.
Its published requirements include Python >=3.9 and `requests>=2.32.2`, whereas
this repository currently pins `requests==2.31.0`. These requirements do not
resolve together unchanged. Do not install the optional backend into an existing
application environment without resolving that pin in the integration step.
Later Datasets releases can raise the Python minimum further. Local Parquet
requires `pyarrow`; saved-HF folders use the Datasets backend too.

Authentication uses Hugging Face's normal environment or saved-login mechanism;
no access token is included in source settings or binding snapshots. No login,
Hub request, or model API call occurs just by importing these modules.

Primary API references:

- [Datasets 4.0 loading and streaming API](https://huggingface.co/docs/datasets/v4.0.0/package_reference/loading_methods)
- [Datasets 4.4.2 declared dependencies](https://github.com/huggingface/datasets/blob/4.4.2/setup.py)
- [Hugging Face repository metadata API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.dataset_info)

## Verification and next boundary

Run the dependency-free foundation checks with:

```sh
python -B -m unittest test_dataset_sources test_dataset_bindings test_dataset_templates
```

Tests use real temporary CSV/TSV/JSON/JSONL files, controlled reader fixtures,
and mocked optional backend APIs. They exercise correlated fields, multiple
sources, twenty concurrent jobs, same-job races, snapshot serialization, source
exhaustion, source edits, metadata-only discovery, and literal substitution.
They do not constitute a live Hub/Arrow check, Windows GUI test, or application
worker/API retry test. Those are subsequent milestones before wiring generation.
