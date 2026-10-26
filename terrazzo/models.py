"""Local model store: an Ollama-like `list / show / rm / import / pull` experience.

Models live in ``$TERRAZZO_MODELS_DIR`` (default ``~/.terrazzo/models``). Each model is
either a directory (MLX/HF snapshot layout) or a symlink to a local directory, so
`import` is cheap and `rm` on a symlink removes only the link.

`pull` downloads Hugging Face snapshots (``mlx-community/...`` repo ids). The Ollama
registry protocol is intentionally NOT implemented: there is no honest way to serve
Ollama tags from an MLX runtime, so the error message says so and points at HF ids.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

RECIPE_FILE = ".terrazzo-recipe.json"


def store_dir() -> Path:
    return Path(os.environ.get("TERRAZZO_MODELS_DIR")
                or (Path.home() / ".terrazzo" / "models"))


def ensure_store_dir() -> Path:
    root = store_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dir_stats(path: Path) -> tuple[int, float]:
    total = 0
    latest = 0.0
    if path.is_symlink():
        target = path.resolve()
        try:
            latest = target.stat().st_mtime
        except OSError:
            latest = 0.0
        path = target
    for p in path.rglob("*"):
        try:
            if not p.is_file():
                continue
            st = p.stat()          # follow symlinks: HF snapshots link to blob files
        except OSError:
            continue
        total += st.st_size
        latest = max(latest, st.st_mtime)
    return total, latest


def _config(path: Path) -> dict:
    cfg = path / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text())
    except Exception:
        return {}


def list_models() -> list[dict]:
    """Entries in the model store (dirs or symlinks), newest first."""
    root = store_dir()
    if not root.is_dir():
        return []
    rows = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() and not entry.is_symlink():
            continue
        size, mtime = _dir_stats(entry)
        cfg = _config(entry)
        rows.append({
            "name": entry.name,
            "path": str(entry.resolve()),
            "size_bytes": size,
            "modified_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime)) if mtime else "-",
            "architecture": cfg.get("architectures", [None])[0] if cfg.get("architectures") else None,
            "quantization": (cfg.get("quantization") or {}).get("group_size") and
                            f"group{cfg['quantization'].get('group_size')}",
            "recipe": (entry / RECIPE_FILE).is_file(),
        })
    rows.sort(key=lambda r: r["modified_at"], reverse=True)
    return rows


def resolve(name_or_path: str) -> Path:
    """Local path (absolute/relative), store entry, or raise with a clear message."""
    p = Path(name_or_path).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p.resolve()
    entry = store_dir() / name_or_path
    if (entry.is_dir() or entry.is_symlink()) and (entry / "config.json").is_file():
        return entry.resolve()
    if "/" in name_or_path and not name_or_path.startswith("."):
        # looks like an HF repo id
        raise FileNotFoundError(
            f"model '{name_or_path}' is not in the local store; run "
            f"`terrazzo pull {name_or_path}` first")
    raise FileNotFoundError(f"model '{name_or_path}' not found (store: {store_dir()})")


def show(name: str) -> dict:
    path = resolve(name)
    cfg = _config(path)
    files = []
    size, mtime = _dir_stats(path)
    for p in sorted(path.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "size_bytes": p.stat().st_size})
    recipe = load_recipe(str(path))
    modelfile = None
    if recipe is not None:
        from .modelfile import render_modelfile

        modelfile = render_modelfile(recipe)
    return {
        "name": Path(name).name,
        "path": str(path),
        "size_bytes": size,
        "files": files,
        "config": {k: cfg.get(k) for k in (
            "architectures", "model_type", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "intermediate_size", "quantization", "tie_word_embeddings") if k in cfg},
        "recipe": recipe,
        "modelfile": modelfile,
    }


def remove(name: str) -> bool:
    entry = store_dir() / name
    if not entry.exists() and not entry.is_symlink():
        return False
    if entry.is_symlink():
        entry.unlink()
    else:
        shutil.rmtree(entry)
    return True


def import_local(path: str, name: str | None = None) -> dict:
    src = Path(path).expanduser().resolve()
    if not (src / "config.json").is_file():
        raise ValueError(f"{src} does not look like a model directory (no config.json)")
    target = ensure_store_dir() / (name or src.name)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; remove it first")
    target.symlink_to(src, target_is_directory=True)
    return {"name": target.name, "path": str(src), "symlink": str(target)}


def _link_model_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    for entry in src.iterdir():
        if entry.name == RECIPE_FILE:
            continue
        (dst / entry.name).symlink_to(entry)


def create(name: str, recipe: dict) -> dict:
    """Create a model recipe (FROM + SYSTEM + PARAMETER) in the local store."""
    if not name or "/" in name or name in (".", ".."):
        raise ValueError("model name must be a simple name (no slashes)")
    base = resolve(recipe["base"])
    target = ensure_store_dir() / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; remove it first")
    _link_model_dir(base, target)
    from .modelfile import to_json

    (target / RECIPE_FILE).write_text(to_json(recipe), encoding="utf-8")
    return {"name": name, "base": str(base), "recipe": recipe,
            "path": str(target)}


def load_recipe(name_or_path: str) -> dict | None:
    try:
        path = resolve(name_or_path)
    except FileNotFoundError:
        return None
    f = path / RECIPE_FILE
    if not f.is_file():
        return None
    from .modelfile import from_json

    return from_json(f.read_text(encoding="utf-8"))


def copy(src: str, dst: str) -> dict:
    """Copy a model entry (symlink: cheap, no duplicated weights)."""
    if not dst or "/" in dst:
        raise ValueError("destination must be a simple name (no slashes)")
    src_path = resolve(src)
    target = ensure_store_dir() / dst
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; remove it first")
    target.symlink_to(src_path, target_is_directory=True)
    return {"source": str(src_path), "destination": dst, "path": str(target)}


def pull(repo_id: str, name: str | None = None) -> dict:
    """Download a Hugging Face snapshot into the store (Ollama registry unsupported)."""
    if ":" in repo_id and "/" not in repo_id:
        raise ValueError(
            "Ollama registry tags are not supported: this runtime serves MLX/GGUF-agnostic "
            "Hugging Face snapshots. Use a repo id such as "
            "'mlx-community/Qwen3-4B-4bit' (see docs/QUICKSTART.zh.md).")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "huggingface_hub is required for `pull` (pip install huggingface_hub)") from exc
    target = ensure_store_dir() / (name or repo_id.split("/")[-1])
    if target.exists():
        raise FileExistsError(f"{target} already exists; remove it first")
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    return {"name": target.name, "path": str(target), "repo_id": repo_id}
