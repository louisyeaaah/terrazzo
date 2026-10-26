"""Local model store: list / show / rm / import / pull error paths."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import models  # noqa: E402


def make_model(dir_path: Path, arch: str = "Qwen3ForCausalLM") -> Path:
    dir_path.mkdir(parents=True)
    (dir_path / "config.json").write_text(json.dumps({
        "architectures": [arch], "model_type": "qwen3", "hidden_size": 2560,
        "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "vocab_size": 151936,
    }))
    (dir_path / "model.safetensors").write_bytes(b"x" * 1024)
    return dir_path


class ModelStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("TERRAZZO_MODELS_DIR")
        os.environ["TERRAZZO_MODELS_DIR"] = str(Path(self.tmp.name) / "store")

    def tearDown(self):
        if self.old is None:
            os.environ.pop("TERRAZZO_MODELS_DIR", None)
        else:
            os.environ["TERRAZZO_MODELS_DIR"] = self.old
        self.tmp.cleanup()

    def test_list_empty_and_after_import(self):
        self.assertEqual(models.list_models(), [])
        src = make_model(Path(self.tmp.name) / "qwen3")
        info = models.import_local(str(src), name="qwen3-imported")
        self.assertEqual(info["name"], "qwen3-imported")
        rows = models.list_models()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "qwen3-imported")
        self.assertGreater(rows[0]["size_bytes"], 0)

    def test_show_returns_config(self):
        src = make_model(Path(self.tmp.name) / "qwen3")
        models.import_local(str(src), name="m1")
        info = models.show("m1")
        self.assertEqual(info["config"]["hidden_size"], 2560)
        self.assertEqual(info["config"]["architectures"], ["Qwen3ForCausalLM"])
        self.assertTrue(any(f["name"] == "model.safetensors" for f in info["files"]))

    def test_rm_symlink_keeps_source(self):
        src = make_model(Path(self.tmp.name) / "qwen3")
        models.import_local(str(src), name="m1")
        self.assertTrue(models.remove("m1"))
        self.assertTrue(src.is_dir())                 # source data untouched
        self.assertEqual(models.list_models(), [])
        self.assertFalse(models.remove("m1"))

    def test_resolve_and_errors(self):
        src = make_model(Path(self.tmp.name) / "qwen3")
        self.assertEqual(models.resolve(str(src)), src.resolve())
        with self.assertRaises(FileNotFoundError):
            models.resolve("missing")
        with self.assertRaises(FileNotFoundError) as ctx:
            models.resolve("mlx-community/Qwen3-4B-4bit")
        self.assertIn("pull", str(ctx.exception))

    def test_create_recipe_and_resolve(self):
        src = make_model(Path(self.tmp.name) / "base")
        models.import_local(str(src), name="base")
        recipe = {"base": "base", "system": "Be terse.",
                  "parameters": {"temperature": 0.3, "max_tokens": 64}}
        info = models.create("custom", recipe)
        self.assertEqual(info["name"], "custom")
        self.assertEqual(models.resolve("custom").name, "custom")
        loaded = models.load_recipe("custom")
        self.assertEqual(loaded["system"], "Be terse.")
        self.assertEqual(loaded["parameters"]["max_tokens"], 64)
        info = models.show("custom")
        self.assertIn("FROM base", info["modelfile"])
        row = [r for r in models.list_models() if r["name"] == "custom"][0]
        self.assertTrue(row["recipe"])

    def test_create_duplicate_raises(self):
        src = make_model(Path(self.tmp.name) / "base")
        models.import_local(str(src), name="base")
        models.create("dup", {"base": "base", "system": None, "parameters": {}})
        with self.assertRaises(FileExistsError):
            models.create("dup", {"base": "base", "system": None, "parameters": {}})

    def test_copy(self):
        src = make_model(Path(self.tmp.name) / "base")
        models.import_local(str(src), name="base")
        info = models.copy("base", "base-copy")
        self.assertEqual(info["destination"], "base-copy")
        self.assertEqual(models.resolve("base-copy"), models.resolve("base"))
        with self.assertRaises(FileExistsError):
            models.copy("base", "base-copy")

    def test_pull_rejects_ollama_tag(self):
        with self.assertRaises(ValueError) as ctx:
            models.pull("qwen3:4b")
        self.assertIn("Hugging Face", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
