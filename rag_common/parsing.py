"""Helpers for cleaning up raw model output (reasoning traces, JSON in prose, ...)."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning)>", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` style reasoning blocks from a completion.

    Also handles templates (e.g. Qwen3, DeepSeek-R1) that put the opening tag in
    the prompt, so the completion only contains the closing ``</think>``.
    """
    if not text:
        return text or ""
    cleaned = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end():]
    return cleaned.strip()


def strip_code_fence(text: str) -> str:
    """Remove a single surrounding Markdown code fence (```json ... ```)."""
    match = _FENCE.match(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _first_balanced(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced ``opener ... closer`` span, ignoring brackets inside strings."""
    start = text.find(opener)
    while start != -1:
        depth, in_str, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find(opener, start + 1)
    return None


def extract_json(text: str, *, repair: bool = True) -> Any:
    """Parse the first JSON object/array found in ``text``.

    Tries, in order: the whole string, the content of a code fence, the first
    balanced ``{...}`` / ``[...]`` span, and finally ``json_repair`` (if
    installed) for truncated or slightly malformed output.

    Raises ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("no text to parse")
    candidates = [text.strip(), strip_code_fence(strip_thinking(text))]
    for cand in list(candidates):
        for opener, closer in (("{", "}"), ("[", "]")):
            span = _first_balanced(cand, opener, closer)
            if span:
                candidates.append(span)
    for cand in candidates:
        if not cand:
            continue
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            pass
    if repair:
        try:
            import json_repair  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            json_repair = None
        if json_repair is not None:
            cand = strip_code_fence(strip_thinking(text))
            start = min((i for i in (cand.find("{"), cand.find("[")) if i != -1), default=-1)
            if start != -1:
                repaired = json_repair.loads(cand[start:])
                if repaired not in ("", None, [], {}):
                    return repaired
    raise ValueError(f"could not find valid JSON in model output: {text[:200]!r}")


def clean_json_text(text: str) -> str:
    """Best effort: return ``text`` reduced to a canonical JSON string, or unchanged."""
    try:
        return json.dumps(extract_json(text), ensure_ascii=False)
    except ValueError:
        return text


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local ``$ref`` (``#/$defs/...``) entries of a JSON schema.

    Pydantic emits ``$defs``/``$ref``; several local inference servers (and
    grammar converters) only understand self-contained schemas.
    Recursive references are left untouched.
    """
    schema = copy.deepcopy(schema)
    defs = {**schema.get("definitions", {}), **schema.get("$defs", {})}

    def _resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                name = ref.split("/")[-1]
                if name in defs and name not in stack:
                    merged = {**copy.deepcopy(defs[name]), **{k: v for k, v in node.items() if k != "$ref"}}
                    return _resolve(merged, stack + (name,))
                return node
            return {k: _resolve(v, stack) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [_resolve(v, stack) for v in node]
        return node

    resolved = _resolve(schema, ())
    if any(isinstance(v, str) and v.startswith("#/") for v in _walk_refs(resolved)):
        # recursive schema: keep the definitions so the remaining refs stay valid
        resolved["$defs"] = defs
    return resolved


def _walk_refs(node: Any):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref":
                yield v
            else:
                yield from _walk_refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_refs(v)


def json_schema_of(response_format: Any) -> dict[str, Any] | None:
    """Return a JSON schema for a pydantic model class / OpenAI-style response_format dict."""
    if response_format is None:
        return None
    if isinstance(response_format, type) and hasattr(response_format, "model_json_schema"):
        return inline_schema_refs(response_format.model_json_schema())
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            js = response_format.get("json_schema") or {}
            schema = js.get("schema", js) if isinstance(js, dict) else None
            return inline_schema_refs(schema) if isinstance(schema, dict) else None
        if "properties" in response_format or response_format.get("type") == "object":
            return inline_schema_refs(response_format)
    return None
