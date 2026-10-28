"""Literal prompt placeholders for the optional dataset integration.

Dataset aliases use ``{{ds:alias}}``. Existing, single-brace placeholders such as
``{context}`` may be supplied separately to :func:`render_template`. Rendering
visits only the original template: neither dataset values nor native values are
treated as templates themselves. No application modules, GUI, or network clients
are imported here.

This renderer is not wired into generation yet. Callers should render authored
templates at the point where both their dataset and stage-specific native values
are available, and must not subsequently run legacy replacement over the result.
Do not render conversation history or generated responses as templates.
"""

import re
from typing import Iterator, Mapping, Optional, Tuple


class DatasetTemplateError(ValueError):
    """An alias or a dataset placeholder cannot be rendered safely."""


_ALIAS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DATASET_PLACEHOLDER = re.compile(r"\{\{ds:([A-Za-z_][A-Za-z0-9_]*)\}\}\Z")
# Whitespace in the reserved namespace is detected so that a mistyped dataset
# placeholder raises an error, instead of silently reaching the model unchanged.
# Native placeholders must have exactly one brace on each side; {{context}}
# therefore cannot be partially replaced by the existing {context} variable.
_TOKEN_START = re.compile(
    r"(?P<dataset>\{\{\s*ds\s*:)|"
    r"(?P<native>(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\}))"
)


def validate_alias(alias: str) -> str:
    """Return ``alias`` unchanged, or raise :class:`DatasetTemplateError`.

    Aliases are case-sensitive ASCII identifiers: a letter or underscore first,
    followed by letters, digits, or underscores. They do not include ``ds:`` or
    braces. Column names are independent and need not follow these restrictions.
    """
    if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
        raise DatasetTemplateError(
            "Dataset aliases must start with a letter or underscore and contain "
            "only letters, digits, or underscores; enter the alias without "
            "'ds:' or braces."
        )
    return alias


def _tokens(template: str) -> Iterator[Tuple[int, int, str, str]]:
    """Yield positions, kind, and name from original template text only."""
    if not isinstance(template, str):
        raise TypeError("A prompt template must be a string.")
    position = 0
    while True:
        match = _TOKEN_START.search(template, position)
        if match is None:
            return
        start = match.start()
        if match.group("dataset") is not None:
            closing = template.find("}}", match.end())
            if closing == -1:
                raise DatasetTemplateError(
                    "Unclosed dataset placeholder at character {}. "
                    "Use {{{{ds:alias}}}}.".format(start)
                )
            end = closing + 2
            placeholder = _DATASET_PLACEHOLDER.fullmatch(template[start:end])
            if placeholder is None:
                raise DatasetTemplateError(
                    "Malformed dataset placeholder at character {}. "
                    "Use {{{{ds:alias}}}} without spaces, with a valid alias.".format(start)
                )
            yield start, end, "dataset", placeholder.group(1)
        else:
            end = match.end()
            yield start, end, "native", template[start + 1:end - 1]
        position = end


def dataset_placeholders(template: str) -> Tuple[str, ...]:
    """List unique dataset aliases in template order without reading any data.

    Malformed reserved placeholders raise :class:`DatasetTemplateError`, just as
    they do during rendering. Other brace syntax and native placeholders are
    ignored. This can be used to check mappings before scheduling generation.
    """
    aliases = []
    seen = set()
    for _, _, kind, name in _tokens(template):
        if kind == "dataset" and name not in seen:
            aliases.append(name)
            seen.add(name)
    return tuple(aliases)


def render_template(
    template: str,
    dataset_values: Mapping[str, str],
    native_values: Optional[Mapping[str, str]] = None,
) -> str:
    """Substitute known placeholders once, without interpreting inserted text.

    ``dataset_values`` maps aliases to already frozen strings. Every dataset
    placeholder must have a mapping; missing mappings and malformed reserved
    syntax raise :class:`DatasetTemplateError`. Native placeholders are replaced
    only when present in ``native_values``. Unknown native placeholders and
    unrelated braces (including ``{{context}}`` and JSON) remain unchanged.

    Referenced values must be strings; empty strings are valid. This intentionally
    does not guess how to serialize lists, dictionaries, numbers, or nulls. The
    binding layer decides their text representation once per generation job.

    Example::

        render_template(
            "Name: {{ds:name}}; passage: {context}",
            {"name": "Ari {context}"},
            {"context": "Literal {{ds:name}} text"},
        )
        # 'Name: Ari {context}; passage: Literal {{ds:name}} text'
    """
    if native_values is None:
        native_values = {}
    pieces = []
    previous = 0
    for start, end, kind, name in _tokens(template):
        pieces.append(template[previous:start])
        if kind == "dataset":
            if name not in dataset_values:
                raise DatasetTemplateError(
                    "No dataset column is mapped to alias {!r} "
                    "({{{{ds:{}}}}}).".format(name, name)
                )
            value = dataset_values[name]
        elif name in native_values:
            value = native_values[name]
        else:
            value = template[start:end]
        if not isinstance(value, str):
            raise DatasetTemplateError(
                "The value for {} placeholder {!r} must already be a string.".format(kind, name)
            )
        pieces.append(value)
        previous = end
    pieces.append(template[previous:])
    return "".join(pieces)
