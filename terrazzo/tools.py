"""OpenAI-style tool/function calling for the local server.

The chat template of Qwen3 (and most mlx-lm chat templates that support tools) teaches
the model to answer function calls as::

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Beijing"}}
    </tool_call>

The template injects the tool schemas; this module validates the request shape and
parses the model output back into OpenAI ``tool_calls``. Streaming is supported by
buffering: the final SSE chunk carries the tool calls (protocol-compatible, coarse
chunking).
"""

from __future__ import annotations

import json
import re
import uuid

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def split_thinking(text: str) -> tuple[str, str]:
    """Split Qwen3 reasoning out of ``text`` -> (thinking, answer).

    Handles completed ``<think>...</think>`` blocks and an unterminated trailing
    block (the model ran out of tokens while still reasoning).
    """
    blocks = [m.group(0)[len("<think>"):-len("</think>")] for m in _THINK_RE.finditer(text)]
    answer = _THINK_RE.sub("", text)
    idx = answer.find("<think>")
    if idx >= 0:
        blocks.append(answer[idx + len("<think>"):])
        answer = answer[:idx]
    return "\n".join(b.strip() for b in blocks if b.strip()), answer.strip()


def strip_thinking(text: str) -> str:
    """Drop Qwen3 ``<think>...</think>`` reasoning from user-visible content."""
    return _THINK_RE.sub("", text).strip()


def validate_tools(tools) -> list[dict]:
    """Return a normalized tool list or raise ValueError with a useful message."""
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list")
    out = []
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{i}] must be an object")
        fn = tool.get("function") or tool
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}].function.name is required")
        params = fn.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"tools[{i}].function.parameters must be an object")
        out.append({"type": "function",
                    "function": {"name": fn["name"],
                                 "description": str(fn.get("description") or ""),
                                 "parameters": params or {"type": "object", "properties": {}}}})
    return out


def normalize_messages(messages) -> list[dict]:
    """Make OpenAI-style messages safe for mlx-lm Jinja templates.

    Jinja chat templates index ``message.content`` directly, so an assistant
    tool-call turn with ``content: null`` (the OpenAI wire format) crashes with
    ``'NoneType' is not iterable``; an empty string is rendered correctly. Tool
    messages always need string content as well.
    """
    out = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        m = dict(msg)
        if m.get("content") is None:
            m["content"] = ""
        elif not isinstance(m["content"], str):
            m["content"] = str(m["content"])
        calls = m.get("tool_calls")
        if calls:
            norm = []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                norm.append({"id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                             "type": "function",
                             "function": {"name": str(fn.get("name", "")),
                                          "arguments": args}})
            m["tool_calls"] = norm
        out.append(m)
    return out


def apply_tool_choice(tools: list[dict], tool_choice) -> tuple[list[dict], list[dict]]:
    """Return (tools passed to the chat template, extra system messages).

    ``tool_choice`` follows the OpenAI shapes; forcing is best-effort because the
    MLX chat templates only accept a tools list (no grammar hook):
      * ``"none"``               -> no tools at all
      * ``"auto"`` / None        -> all tools
      * ``{"function": {...}}``  -> only that tool + an explicit instruction
      * ``"required"``           -> all tools + an explicit instruction
    """
    if tool_choice in (None, "auto"):
        return tools, []
    if tool_choice == "none":
        return [], []
    if tool_choice == "required":
        return tools, [{"role": "system",
                        "content": "You must call one of the provided tools before answering."}]
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") or tool_choice.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.function.name is required")
        picked = [t for t in tools if t["function"]["name"] == name]
        if not picked:
            raise ValueError(f"tool_choice names unknown function: {name}")
        return picked, [{"role": "system",
                         "content": f"You must call the tool `{name}` before answering."}]
    raise ValueError(f"unsupported tool_choice: {tool_choice!r}")


def _as_call(payload: dict) -> dict | None:
    name = payload.get("name") or payload.get("function")
    if isinstance(name, dict):
        name = name.get("name")
    args = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        args = {"value": args}
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split ``<tool_call>`` blocks out of ``text`` -> (remaining text, tool calls)."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        call = _as_call(payload) if isinstance(payload, dict) else None
        if call is not None:
            calls.append(call)
    # drop exact duplicates (models sometimes emit the same call twice)
    seen = set()
    unique = []
    for call in calls:
        key = (call["function"]["name"], call["function"]["arguments"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(call)
    cleaned = strip_thinking(_TOOL_CALL_RE.sub("", text))
    if unique:
        return cleaned, unique
    # fallback: a bare JSON object (some templates/models emit it without tags)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return text, []
        call = _as_call(payload) if isinstance(payload, dict) else None
        if call is not None:
            return "", [call]
    return text, []
