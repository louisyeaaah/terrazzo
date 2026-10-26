"""Schema-constrained JSON decoding.

Guarantees the decoded text is valid JSON that satisfies the supported JSON-Schema
subset: ``type`` (object/array/string/number/integer/boolean/null), ``properties``,
``required``, ``additionalProperties`` (false), ``items``, ``enum``, ``const``.

The trick is a *prefix* validator: every candidate token is appended to the text so
far and classified as ``invalid`` / ``prefix`` / ``complete``; only tokens that keep
the text a valid prefix are allowed, and generation stops as soon as the document is
complete. That makes the output valid by construction (no retry/repair loop).

Not supported (rejected up front): ``$ref``, ``allOf``/``anyOf``/``oneOf``,
``patternProperties``, ``format``. They raise :class:`SchemaError` (-> HTTP 400).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


class SchemaError(ValueError):
    """The schema is invalid or uses unsupported features."""


_UNSUPPORTED = ("$ref", "allOf", "anyOf", "oneOf", "patternProperties", "not", "if")
_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_NUMBER_PREFIX_RE = re.compile(r"-?(?:0|[1-9][0-9]*)?(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?$")


def compile_schema(schema) -> dict:
    """Validate the schema and normalise it (raises SchemaError on problems)."""
    if not isinstance(schema, dict):
        raise SchemaError("schema must be a JSON object")
    for key in _UNSUPPORTED:
        if key in schema:
            raise SchemaError(f"unsupported schema keyword: {key}")
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else list(types)
        for t in types:
            if t not in JSON_TYPES:
                raise SchemaError(f"unsupported type: {t!r}")
        if len(types) == 1:
            schema = {**schema, "type": types[0]}
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            raise SchemaError("properties must be an object")
        schema = {**schema, "properties": {k: compile_schema(v) for k, v in props.items()}}
    if "items" in schema:
        schema = {**schema, "items": compile_schema(schema["items"])}
    if "required" in schema and not isinstance(schema["required"], list):
        raise SchemaError("required must be an array of property names")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        schema = {**schema, "additionalProperties": compile_schema(schema["additionalProperties"])}
    return schema


def _types_of(schema: dict) -> set[str]:
    t = schema.get("type")
    if t is None:
        return JSON_TYPES
    return {t} if isinstance(t, str) else set(t)


class _Invalid(Exception):
    pass


class _Incomplete(Exception):
    pass


@dataclass
class _Parser:
    text: str
    schema: dict
    pos: int = 0

    def _peek(self) -> str:
        if self.pos >= len(self.text):
            raise _Incomplete()
        return self.text[self.pos]

    def _ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\n\r":
            self.pos += 1

    def document(self) -> str:
        self._ws()
        self.value(self.schema)
        self._ws()
        if self.pos < len(self.text):
            raise _Invalid("trailing characters")
        return "complete"

    def _expect(self, ch: str) -> None:
        if self.pos >= len(self.text):
            raise _Incomplete()
        if self.text[self.pos] != ch:
            raise _Invalid(f"expected {ch!r}")
        self.pos += 1

    def _literal(self, word: str) -> None:
        for i, ch in enumerate(word):
            if self.pos + i >= len(self.text):
                raise _Incomplete()
            if self.text[self.pos + i] != ch:
                raise _Invalid(f"expected {word!r}")
        self.pos += len(word)

    def value(self, schema: dict) -> None:
        self._ws()
        if self.pos >= len(self.text):
            raise _Incomplete()
        ch = self.text[self.pos]
        types = _types_of(schema)
        if "const" in schema or "enum" in schema:
            options = [schema["const"]] if "const" in schema else list(schema["enum"])
            self._literal_option(options)
            return
        if ch == '"':
            if "string" not in types:
                raise _Invalid("string not allowed")
            self.string()
            return
        if ch == "{":
            if "object" not in types:
                raise _Invalid("object not allowed")
            self.object(schema)
            return
        if ch == "[":
            if "array" not in types:
                raise _Invalid("array not allowed")
            self.array(schema)
            return
        if ch == "t":
            if "boolean" not in types:
                raise _Invalid("boolean not allowed")
            self._literal("true")
            return
        if ch == "f":
            if "boolean" not in types:
                raise _Invalid("boolean not allowed")
            self._literal("false")
            return
        if ch == "n":
            if "null" not in types:
                raise _Invalid("null not allowed")
            self._literal("null")
            return
        if ch == "-" or ch.isdigit():
            if not (types & {"number", "integer"}):
                raise _Invalid("number not allowed")
            self.number(types)
            return
        raise _Invalid(f"unexpected character {ch!r}")

    def _literal_option(self, options: list) -> None:
        """Match the prefix against json-encoded enum/const options."""
        rest = self.text[self.pos:]
        candidates = [json.dumps(o, ensure_ascii=False) for o in options]
        for cand in candidates:
            if cand.startswith(rest) and len(rest) < len(cand):
                raise _Incomplete()
            if rest.startswith(cand):
                self.pos += len(cand)
                return
        raise _Invalid("not one of the allowed values")

    def string(self) -> None:
        start = self.pos
        self._expect('"')
        while True:
            if self.pos >= len(self.text):
                raise _Incomplete()
            ch = self.text[self.pos]
            if ch == '"':
                self.pos += 1
                return
            if ch == "\\":
                self.pos += 1
                if self.pos >= len(self.text):
                    raise _Incomplete()
                esc = self.text[self.pos]
                if esc in '"\\/bfnrt':
                    self.pos += 1
                elif esc == "u":
                    for _ in range(4):
                        self.pos += 1
                        if self.pos >= len(self.text):
                            raise _Incomplete()
                        if not re.match(r"[0-9a-fA-F]", self.text[self.pos]):
                            raise _Invalid("bad unicode escape")
                    self.pos += 1
                else:
                    raise _Invalid("bad escape")
            elif ord(ch) < 0x20:
                raise _Invalid("control character in string")
            else:
                self.pos += 1

    def number(self, types: set[str]) -> None:
        rest = self.text[self.pos:]
        i = 0
        while i < len(rest) and rest[i] in "-+.eE0123456789":
            i += 1
        chunk = rest[:i]
        if chunk and _NUMBER_RE.match(chunk):
            # a complete number could still be extended (e.g. "1" -> "12")
            self.pos += i
            return
        if chunk and _NUMBER_PREFIX_RE.match(chunk):
            self.pos += i
            return
        if rest:
            raise _Invalid("bad number")
        raise _Incomplete()

    def object(self, schema: dict) -> None:
        self._expect("{")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional = schema.get("additionalProperties", True)
        seen: set[str] = set()
        self._ws()
        if self.pos >= len(self.text):
            raise _Incomplete()
        if self.text[self.pos] == "}":
            self.pos += 1
            missing = required - seen
            if missing:
                raise _Invalid(f"missing required properties: {sorted(missing)}")
            return
        while True:
            self._ws()
            key = self._object_key(props, additional)
            seen.add(key)
            self._ws()
            self._expect(":")
            child = props.get(key)
            if child is None:
                if additional is True:
                    child = {}
                elif isinstance(additional, dict):
                    child = additional
                else:
                    raise _Invalid(f"additional property {key!r} not allowed")
            self.value(child)
            self._ws()
            if self.pos >= len(self.text):
                raise _Incomplete()
            ch = self.text[self.pos]
            if ch == ",":
                self.pos += 1
                continue
            if ch == "}":
                self.pos += 1
                missing = required - seen
                if missing:
                    raise _Invalid(f"missing required properties: {sorted(missing)}")
                return
            raise _Invalid("expected , or }")

    def _object_key(self, props: dict, additional) -> str:
        if self.pos >= len(self.text) or self.text[self.pos] != '"':
            # allow a partial key start: quote required
            raise _Incomplete() if self.pos >= len(self.text) else _Invalid("expected key")
        start = self.pos
        self.string()
        raw = self.text[start:self.pos]
        try:
            key = json.loads(raw)
        except Exception:
            raise _Invalid("bad key") from None
        if additional is False and key not in props:
            raise _Invalid(f"property {key!r} not allowed")
        if key in props:
            return key
        if isinstance(additional, dict):
            return key
        return key  # additional allowed (or unknown key with additional schema)

    def array(self, schema: dict) -> None:
        self._expect("[")
        items = schema.get("items", {})
        self._ws()
        if self.pos >= len(self.text):
            raise _Incomplete()
        if self.text[self.pos] == "]":
            self.pos += 1
            return
        while True:
            self.value(items)
            self._ws()
            if self.pos >= len(self.text):
                raise _Incomplete()
            ch = self.text[self.pos]
            if ch == ",":
                self.pos += 1
                continue
            if ch == "]":
                self.pos += 1
                return
            raise _Invalid("expected , or ]")


class JsonConstraint:
    """Classify a text prefix as invalid / prefix / complete for a schema."""

    def __init__(self, schema: dict | None = None):
        self.schema = compile_schema(schema or {})

    def classify(self, text: str) -> str:
        p = _Parser(text, self.schema)
        try:
            return p.document()
        except _Incomplete:
            return "prefix"
        except (_Invalid, SchemaError):
            return "invalid"


def json_generate(model, tokenizer, prompt_ids: list[int], schema: dict,
                  *, max_tokens: int = 512, top_k: int = 128,
                  temperature: float = 0.0) -> tuple[str, dict]:
    """Constrained greedy decoding of a JSON document matching ``schema``."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    constraint = JsonConstraint(schema)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    prefill = list(prompt_ids[:-1])
    chunk = 1024
    for i in range(0, len(prefill), chunk):
        logits = model(mx.array([prefill[i : i + chunk]]), cache=cache)
        mx.eval(logits)
    ttft = time.perf_counter() - t0
    pending = prompt_ids[-1]
    text = ""
    steps = 0
    complete = False
    while steps < max_tokens:
        logits = model(mx.array([[pending]]), cache=cache)
        mx.eval(logits)
        steps += 1
        row = logits[0, -1]
        vocab = row.shape[0]
        order = None
        chosen = None
        for k in (top_k, 2048, vocab):
            k = min(k, vocab)
            if order is None:
                order = mx.argsort(row)[::-1]
                mx.eval(order)
            for tid in order[:k].tolist():
                piece = tokenizer.decode([int(tid)], skip_special_tokens=True)
                if not piece:
                    continue
                status = constraint.classify(text + piece)
                if status == "invalid":
                    continue
                chosen = (int(tid), piece, status)
                break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("no schema-valid token found in vocabulary")
        tid, piece, status = chosen
        text += piece
        pending = tid
        if status == "complete":
            complete = True
            break
    total = time.perf_counter() - t0
    stats = {
        "steps": steps,
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "complete": complete,
        "text_chars": len(text),
    }
    return text, stats
