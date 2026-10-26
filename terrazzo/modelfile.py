"""Ollama-style Modelfile parsing (supported subset).

Supported directives (one per line, or a triple-quoted block for SYSTEM):

    FROM <store-name | /path/to/mlx-model>
    SYSTEM <text>
    PARAMETER temperature 0.7
    PARAMETER top_p 0.9
    PARAMETER top_k 20
    PARAMETER num_predict 256
    PARAMETER stop "<|im_end|>"        (repeats allowed)

TEMPLATE, ADAPTER and MESSAGE are rejected explicitly: MLX models carry their own
chat template and this runtime does not apply Jinja templates from a Modelfile, so
pretending to support them would silently change model behaviour. LICENSE is accepted
and ignored (metadata only).
"""

from __future__ import annotations

import json
import shlex

PARAM_ALIASES = {
    "num_predict": "max_tokens",
    "max_tokens": "max_tokens",
    "num_ctx": "num_ctx",          # accepted; the server uses its own context limit
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "stop": "stop",
}

_SUPPORTED = {"FROM", "SYSTEM", "PARAMETER", "LICENSE"}
_UNSUPPORTED = {"TEMPLATE", "ADAPTER", "MESSAGE", "TOKENIZER"}
_TRIPLE = '"""'


def _value_after(directive: str, rest: str, lines: list[str], index: int) -> tuple[str, int]:
    """Parse a directive value starting at ``rest``; supports multi-line triple quotes."""
    rest = rest.strip()
    if rest.startswith(_TRIPLE):
        body = rest[len(_TRIPLE):]
        if body.rstrip().endswith(_TRIPLE):
            return body.rstrip()[:-len(_TRIPLE)], index
        chunks = [body]
        i = index + 1
        while i < len(lines):
            line = lines[i]
            if line.rstrip().endswith(_TRIPLE):
                chunks.append(line.rstrip()[:-len(_TRIPLE)])
                return "\n".join(chunks), i
            chunks.append(line)
            i += 1
        raise ValueError(f"{directive}: unterminated \"\"\" block")
    if rest.startswith('"') or rest.startswith("'"):
        try:
            parts = shlex.split(rest)
        except ValueError as exc:
            raise ValueError(f"{directive}: {exc}") from exc
        return (parts[0] if parts else ""), index
    return rest, index


def _parse_scalar(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_modelfile(text: str) -> dict:
    lines = text.splitlines()
    recipe: dict = {"base": None, "system": None, "parameters": {}}
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        directive = parts[0].upper()
        rest = parts[1] if len(parts) > 1 else ""
        if directive in _UNSUPPORTED:
            raise ValueError(
                f"Modelfile directive {directive} is not supported: MLX models carry "
                f"their own chat template; supported directives are FROM, SYSTEM, "
                f"PARAMETER, LICENSE")
        if directive not in _SUPPORTED:
            raise ValueError(f"unknown Modelfile directive: {parts[0]}")
        if directive == "LICENSE":
            continue
        value, i = _value_after(directive, rest, lines, i - 1)
        i += 1
        if directive == "FROM":
            if not value.strip():
                raise ValueError("FROM needs a model name or path")
            recipe["base"] = value.strip()
        elif directive == "SYSTEM":
            recipe["system"] = value.strip("\n")
        else:  # PARAMETER
            sub = value.split(None, 1)
            if not sub:
                raise ValueError("PARAMETER needs a name and a value")
            name = sub[0].lower()
            if name not in PARAM_ALIASES:
                raise ValueError(
                    f"unsupported PARAMETER {sub[0]!r} (supported: "
                    f"{', '.join(sorted(set(PARAM_ALIASES)))})")
            raw_val = sub[1] if len(sub) > 1 else ""
            try:
                pieces = shlex.split(raw_val)
                raw_val = " ".join(pieces) if pieces else raw_val
            except ValueError:
                raw_val = raw_val.strip('"\'')
            val = _parse_scalar(raw_val) if raw_val != "" else ""
            key = PARAM_ALIASES[name]
            if key == "stop":
                recipe["parameters"].setdefault("stop", []).append(str(val))
            else:
                recipe["parameters"][key] = val
    if not recipe["base"]:
        raise ValueError("Modelfile requires FROM <model>")
    return recipe


def render_modelfile(recipe: dict) -> str:
    out = [f"FROM {recipe['base']}"]
    if recipe.get("system"):
        out += ["", 'SYSTEM """', recipe["system"], '"""']
    params = recipe.get("parameters") or {}
    for key, value in params.items():
        if key == "stop":
            for item in value:
                out.append(f'PARAMETER stop "{item}"')
            continue
        name = "num_predict" if key == "max_tokens" else key
        out.append(f"PARAMETER {name} {value}")
    return "\n".join(out) + "\n"


def to_json(recipe: dict) -> str:
    return json.dumps(recipe, ensure_ascii=False, sort_keys=True)


def from_json(text: str) -> dict:
    data = json.loads(text)
    if not isinstance(data, dict) or not data.get("base"):
        raise ValueError("recipe is missing FROM/base")
    return data