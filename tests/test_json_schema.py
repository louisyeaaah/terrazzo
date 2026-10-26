"""Schema-constrained JSON decoding: prefix validator + end-to-end generation."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.json_schema import JsonConstraint, SchemaError, compile_schema  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


class PrefixConstraintTest(unittest.TestCase):
    def setUp(self):
        self.c = JsonConstraint(SCHEMA)

    def test_prefixes(self):
        for text in ["", "{", '{"', '{"name":', '{"name": "Al', '{"name": "Alice",',
                     '{"name": "Alice", "age": 3']:
            self.assertEqual(self.c.classify(text), "prefix", text)

    def test_complete(self):
        for text in ['{"name": "Alice", "age": 3}',
                     '{"name": "A", "age": 3, "tags": ["x"]}']:
            self.assertEqual(self.c.classify(text), "complete", text)

    def test_invalid(self):
        for text in ["xxx", '{"name": 1', '{"x": 1}', '{"name": "A", "age": 3, "y": 2}']:
            self.assertEqual(self.c.classify(text), "invalid", text)

    def test_required_before_close(self):
        self.assertEqual(self.c.classify('{"name": "A"}'), "invalid")

    def test_enum_and_const(self):
        c = JsonConstraint({"type": "object",
                            "properties": {"color": {"enum": ["red", "green"]},
                                           "n": {"const": 42}},
                            "required": ["color", "n"],
                            "additionalProperties": False})
        self.assertEqual(c.classify('{"color": "r'), "prefix")
        self.assertEqual(c.classify('{"color": "blue"'), "invalid")
        self.assertEqual(c.classify('{"color": "red", "n": 4'), "prefix")
        self.assertEqual(c.classify('{"color": "red", "n": 42}'), "complete")

    def test_unsupported_schema_rejected(self):
        for bad in ({"$ref": "#/x"}, {"type": "frobnicate"}, {"type": "object", "allOf": []}):
            with self.assertRaises(SchemaError):
                compile_schema(bad)

    def test_every_prefix_of_valid_doc_is_classified(self):
        doc = '{"name": "Alice", "age": 3, "tags": ["a", "b"]}'
        for i in range(len(doc)):
            self.assertIn(self.c.classify(doc[:i]), ("prefix", "complete"), doc[:i])
        self.assertEqual(self.c.classify(doc), "complete")


class JsonGenerateModelTest(unittest.TestCase):
    def test_generate_conforming_json(self):
        model = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model or not Path(model).is_dir():
            self.skipTest("set TERRAZZO_MAC_MODEL")
        from mlx_lm import load

        from terrazzo.json_schema import json_generate

        mdl, tok = load(model)
        schema = {"type": "object",
                  "properties": {"name": {"type": "string"},
                                 "age": {"type": "integer"},
                                 "city": {"type": "string"}},
                  "required": ["name", "age"], "additionalProperties": False}
        prompt = tok.encode("请给我一个虚构人物的 JSON，包含 name、age、city")
        text, stats = json_generate(mdl, tok, prompt, schema, max_tokens=256)
        parsed = json.loads(text)
        self.assertIn("name", parsed)
        self.assertIn("age", parsed)
        self.assertIsInstance(parsed["age"], int)
        self.assertTrue(stats["complete"])


if __name__ == "__main__":
    unittest.main()
