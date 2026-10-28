"""Focused, dependency-free tests: python -m unittest test_dataset_templates."""

import unittest

from dataset_templates import (
    DatasetTemplateError,
    dataset_placeholders,
    render_template,
    validate_alias,
)


class DatasetTemplateTests(unittest.TestCase):
    def test_dataset_and_native_aliases_are_distinct(self):
        self.assertEqual(
            render_template(
                "{{ds:context}} / {context} / {{context}} / {unknown}",
                {"context": "dataset"},
                {"context": "passage"},
            ),
            "dataset / passage / {{context}} / {unknown}",
        )

    def test_inserted_values_remain_opaque_in_both_directions(self):
        dataset = '{context} {{ds:other}} "quoted" \\path\nnext'
        native = '{{ds:name}} {context} \\other\nline'
        self.assertEqual(
            render_template(
                "{{ds:name}}\n{context}\n{{ds:other}}",
                {"name": dataset, "other": "resolved only here"},
                {"context": native},
            ),
            dataset + "\n" + native + "\nresolved only here",
        )

    def test_native_values_do_not_expand_each_other(self):
        self.assertEqual(
            render_template("{subject} {context}", {}, {"subject": "{context}", "context": "text"}),
            "{context} text",
        )

    def test_repeated_aliases_and_empty_strings(self):
        self.assertEqual(
            render_template("{{ds:a}}{{ds:empty}}/{{ds:a}}", {"a": "same", "empty": ""}),
            "same/same",
        )

    def test_unrelated_json_code_tags_pipes_and_braces_survive(self):
        template = '{"name": "{{ds:name}}", "options": {"enabled": true}}\n{a: 1}\n<tag>|pipe|{{other}}'
        self.assertEqual(
            render_template(template, {"name": "Ari"}),
            '{"name": "Ari", "options": {"enabled": true}}\n{a: 1}\n<tag>|pipe|{{other}}',
        )

    def test_empty_mapping_preserves_plain_and_native_templates(self):
        template = "Plain {unmapped}; {{context}}; {}."
        self.assertEqual(render_template(template, {}), template)
        self.assertEqual(render_template("", {}), "")
        self.assertEqual(render_template("{context}", {}, {"context": ""}), "")

    def test_unknown_alias_fails_without_exposing_values(self):
        with self.assertRaisesRegex(DatasetTemplateError, "missing") as caught:
            render_template("{{ds:missing}}", {"other": "sensitive row content"})
        self.assertNotIn("sensitive row content", str(caught.exception))

    def test_malformed_reserved_syntax_fails(self):
        malformed = (
            "{{ds:}}",
            "{{ds:has-hyphen}}",
            "{{ds:1first}}",
            "{{ds: has_space}}",
            "{{ ds:name}}",
            "{{ds :name}}",
            "{{ds:name }}",
            "{{ds:name}",
            "{{ds:name",
            "{{ds:{context}}}",
        )
        for template in malformed:
            with self.subTest(template=template):
                with self.assertRaises(DatasetTemplateError):
                    render_template(template, {"name": "Ari"})
                with self.assertRaises(DatasetTemplateError):
                    dataset_placeholders(template)

    def test_placeholder_inventory_preserves_first_occurrence_order(self):
        self.assertEqual(
            dataset_placeholders("{{ds:b}} {context} {{ds:a}} {{ds:b}} {{other}}"),
            ("b", "a"),
        )
        self.assertEqual(dataset_placeholders("{context} plain {{other}}"), ())

    def test_alias_validation_does_not_rewrite_input(self):
        for alias in ("name", "Name", "_name", "name_2", "context"):
            self.assertEqual(validate_alias(alias), alias)
        for alias in ("", "has space", "has-hyphen", "2name", " name", "name ", "ds:name", "{{ds:name}}", "é", None):
            with self.subTest(alias=alias):
                with self.assertRaises(DatasetTemplateError):
                    validate_alias(alias)

    def test_values_must_have_been_formatted_by_the_binding_layer(self):
        for value in (None, 3, ["a"], {"key": "value"}):
            with self.subTest(value=value):
                with self.assertRaises(DatasetTemplateError):
                    render_template("{{ds:value}}", {"value": value})
                with self.assertRaises(DatasetTemplateError):
                    render_template("{context}", {}, {"context": value})

    def test_template_must_be_a_string(self):
        with self.assertRaises(TypeError):
            render_template(None, {})
        with self.assertRaises(TypeError):
            dataset_placeholders(None)


if __name__ == "__main__":
    unittest.main()
