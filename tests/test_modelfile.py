"""Modelfile parsing/rendering (Ollama-compatible subset)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.modelfile import (from_json, parse_modelfile,  # noqa: E402
                                    render_modelfile, to_json)


class ModelfileTest(unittest.TestCase):
    def test_parse_full(self):
        recipe = parse_modelfile(
            'FROM mlx-community/Qwen3-4B-4bit\n'
            'SYSTEM """\nYou are terse.\n"""\n'
            'PARAMETER temperature 0.3\n'
            'PARAMETER top_p 0.9\n'
            'PARAMETER num_predict 128\n'
            'PARAMETER stop "<|im_end|>"\n'
            'PARAMETER stop "<|endoftext|>"\n'
            'LICENSE Apache-2.0\n')
        self.assertEqual(recipe["base"], "mlx-community/Qwen3-4B-4bit")
        self.assertEqual(recipe["system"], "You are terse.")
        self.assertEqual(recipe["parameters"]["temperature"], 0.3)
        self.assertEqual(recipe["parameters"]["max_tokens"], 128)
        self.assertEqual(recipe["parameters"]["stop"], ["<|im_end|>", "<|endoftext|>"])

    def test_single_line_system_and_quoted_number(self):
        recipe = parse_modelfile('FROM x\nSYSTEM "Be brief"\nPARAMETER temperature "0.5"\n')
        self.assertEqual(recipe["system"], "Be brief")
        self.assertEqual(recipe["parameters"]["temperature"], 0.5)

    def test_missing_from(self):
        with self.assertRaises(ValueError):
            parse_modelfile("PARAMETER temperature 0.2\n")

    def test_unsupported_template_is_explicit(self):
        with self.assertRaises(ValueError) as ctx:
            parse_modelfile('FROM x\nTEMPLATE "{{ .Prompt }}"\n')
        self.assertIn("TEMPLATE", str(ctx.exception))

    def test_unknown_directive(self):
        with self.assertRaises(ValueError):
            parse_modelfile("FROM x\nBOGUS y\n")

    def test_unknown_parameter(self):
        with self.assertRaises(ValueError):
            parse_modelfile("FROM x\nPARAMETER repeat_penalty 1.1\n")

    def test_render_round_trip(self):
        recipe = {"base": "m", "system": "Be brief",
                  "parameters": {"temperature": 0.2, "max_tokens": 64,
                                 "stop": ["<e>"]}}
        text = render_modelfile(recipe)
        self.assertIn("FROM m", text)
        self.assertIn("PARAMETER num_predict 64", text)
        self.assertIn('PARAMETER stop "<e>"', text)
        parsed = parse_modelfile(text)
        self.assertEqual(parsed["system"], "Be brief")
        self.assertEqual(parsed["parameters"]["max_tokens"], 64)
        self.assertEqual(parsed["parameters"]["stop"], ["<e>"])

    def test_json_round_trip(self):
        recipe = {"base": "m", "system": None, "parameters": {"temperature": 0.1}}
        self.assertEqual(from_json(to_json(recipe)), recipe)


if __name__ == "__main__":
    unittest.main()
