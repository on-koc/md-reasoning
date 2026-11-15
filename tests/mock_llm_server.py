#!/usr/bin/env python
"""Offline mock of an LLM + embedding server, for smoke-testing the indexing scripts.

Speaks both protocols the scripts use:
  * OpenAI-compatible:  GET /v1/models, POST /v1/chat/completions, POST /v1/embeddings
  * Ollama native:      GET /api/version, GET /api/tags, POST /api/chat, POST /api/embed

Chat answers are produced by a tiny rule-based "model" that recognises the
extraction prompts of MS GraphRAG, LightRAG and HippoRAG 2 and answers in the
format each one parses (capitalised phrases become entities, co-occurrence in a
sentence becomes a relation). Embeddings are hashed bags of words, so texts that
share words get similar vectors.

    python tests/mock_llm_server.py --port 11434
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STOP_WORDS = {
    "The", "A", "An", "In", "On", "At", "Its", "It", "She", "He", "They", "These", "This", "Those", "Data",
    "Dr", "Mr", "Ms", "Measurements", "Respond", "Text", "Entity", "Output", "Example", "Paragraph", "Return",
}
ORG_HINTS = ("Institute", "University", "Consortium", "Authority", "Council", "Survey", "Prize", "Company")
GEO_HINTS = ("Range", "Glacier", "River", "City", "Valley", "Valdoria", "Halvik", "Mountains", "Lake")


# ---------------------------------------------------------------------------
# the rule-based "model"
# ---------------------------------------------------------------------------


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def entities_in(sentence: str) -> list[str]:
    found = []
    for match in re.finditer(r"\b[A-Z][\w-]*(?:\s+(?:of\s+)?[A-Z][\w-]*)*", sentence):
        words = match.group(0).split()
        while words and words[0] in STOP_WORDS:
            words = words[1:]
        name = " ".join(words).strip()
        if len(name) > 2 and name not in found and not name.isupper():
            found.append(name)
    return found


def entity_type(name: str) -> str:
    if any(h in name for h in ORG_HINTS):
        return "organization"
    if any(h in name for h in GEO_HINTS):
        return "geo"
    if len(name.split()) == 2:
        return "person"
    return "event" if any(ch.isdigit() for ch in name) else "concept"


def extract(text: str, max_entities: int = 12) -> tuple[dict[str, str], list[tuple[str, str, str, str]]]:
    """Return ``{entity: description}`` and ``[(src, tgt, predicate, sentence)]``."""
    ents: dict[str, str] = {}
    rels: list[tuple[str, str, str, str]] = []
    for sent in sentences(text):
        names = [n for n in entities_in(sent)]
        for n in names:
            if n not in ents and len(ents) < max_entities:
                ents[n] = sent
        names = [n for n in names if n in ents]
        for a, b in zip(names, names[1:]):
            between = sent.split(a, 1)[-1].split(b, 1)[0]
            pred = " ".join(re.findall(r"[a-zA-Z]+", between)[:6]) or "related to"
            rels.append((a, b, pred, sent))
    return ents, rels


def graphrag_extraction(text: str) -> str:
    ents, rels = extract(text)
    rows = [f'("entity"<|>{n.upper()}<|>{entity_type(n).upper()}<|>{n} is mentioned as follows: {d})' for n, d in ents.items()]
    rows += [f'("relationship"<|>{a.upper()}<|>{b.upper()}<|>{a} {p} {b}<|>{5})' for a, b, p, _ in rels]
    return "\n##\n".join(rows) + "\n<|COMPLETE|>"


def lightrag_extraction(text: str, as_json: bool) -> str:
    ents, rels = extract(text)
    if as_json:
        return json.dumps(
            {
                "entities": [{"name": n, "type": entity_type(n), "description": d} for n, d in ents.items()],
                "relationships": [
                    {"source": a, "target": b, "keywords": p, "description": s} for a, b, p, s in rels
                ],
            }
        )
    rows = [f"entity<|#|>{n}<|#|>{entity_type(n)}<|#|>{d}" for n, d in ents.items()]
    rows += [f"relation<|#|>{a}<|#|>{b}<|#|>{p}<|#|>{s}" for a, b, p, s in rels]
    return "\n".join(rows) + "\n<|COMPLETE|>"


def community_report(prompt: str) -> str:
    names = []
    for line in prompt.splitlines():
        parts = line.split(",")
        if len(parts) >= 3 and parts[0].strip().isdigit():
            names.append(parts[1].strip())
    names = names[:5] or ["the community"]
    return json.dumps(
        {
            "title": f"{names[0]} and related entities",
            "summary": f"This community centres on {', '.join(names)}.",
            "rating": 5.0,
            "rating_explanation": "Moderate importance (mock rating).",
            "findings": [{"summary": f"{n} is a member of this community", "explanation": f"{n} appears in the data."} for n in names[:3]],
        }
    )


def schema_stub(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {k: schema_stub(v) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind in ("number", "integer"):
        return 1
    if kind == "boolean":
        return True
    return "mock"


def respond(messages: list[dict[str, Any]], json_format: Any) -> str:
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    users = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    last = users[-1] if users else ""
    prompt = "\n".join(str(m.get("content", "")) for m in messages)

    # --- connectivity checks used by the scripts ---
    if "single word OK" in last or "exactly one word: OK" in last:
        return "OK"
    if "capital city of France" in last:
        return json.dumps({"city": "Paris", "country": "France"})

    # --- HippoRAG 2 ---
    if "extract named entities from the given paragraph" in system:
        ents, _ = extract(last)
        return json.dumps({"named_entities": list(ents)})
    if last.startswith("Convert the paragraph into a JSON dict"):
        m = re.search(r"Paragraph:\s*```(.*?)```", last, re.S)
        _, rels = extract(m.group(1) if m else last)
        return json.dumps({"triples": [[a, p, b] for a, b, p, _ in rels]})

    # --- MS GraphRAG ---
    if "-Real Data-" in last and "Entity_types:" in last:
        part = last.split("-Real Data-")[-1]
        m = re.search(r"Text:\s*(.*?)\n#{5,}\s*\nOutput:", part, re.S)
        return graphrag_extraction(m.group(1) if m else part)
    if last.startswith("MANY entities and relationships were missed"):
        return "<|COMPLETE|>"
    if "Answer Y if there are still entities" in last:
        return "N"
    if "Description List:" in last and "Entities:" in last:
        descs = re.findall(r'"((?:[^"\\]|\\.)*)"', last.split("Description List:")[-1])
        return " ".join(dict.fromkeys(descs))[:1500] or "No description."
    if json_format is not None and ("community" in prompt.lower() and "findings" in prompt.lower()):
        return community_report(prompt)

    # --- LightRAG ---
    if "---Input Text---" in last:
        m = re.search(r"---Input Text---\s*```\s*(.*?)```", last, re.S)
        text = m.group(1) if m else last
        return lightrag_extraction(text, as_json='"entities"' in system or json_format is not None)
    if "Based on the last extraction task" in last:
        return '{"entities": [], "relationships": []}' if json_format is not None else "<|COMPLETE|>"
    if "Description List:" in last and "---Input---" in last:
        body = last.split("Description List:")[-1]
        descs = re.findall(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)"', body) or [body.strip()[:500]]
        return " ".join(dict.fromkeys(descs))[:1500]
    if "high_level_keywords" in prompt:
        ents, _ = extract(last)
        return json.dumps({"high_level_keywords": ["overview"], "low_level_keywords": list(ents)[:5]})

    if isinstance(json_format, dict):
        return json.dumps(schema_stub(json_format))
    if json_format is not None:
        return json.dumps({"answer": "mock"})
    return "This is a mock answer."


def embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "MockLLM/1.0"
    dim = 64
    reject_json_schema = False
    latency = 0.0
    counter = {"chat": 0, "embed": 0}
    lock = threading.Lock()

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model", "owned_by": "mock"}]})
        elif self.path.rstrip("/") == "/api/version":
            self._send(200, {"version": "0.0.0-mock"})
        elif self.path.rstrip("/") == "/api/tags":
            self._send(200, {"models": [{"name": "mock", "model": "mock"}]})
        elif self.path.rstrip("/") == "/stats":
            self._send(200, self.counter)
        else:
            self._send(200, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        if self.latency:
            time.sleep(self.latency)
        try:
            body = self._body()
            path = self.path.rstrip("/")
            if path == "/v1/chat/completions":
                self._openai_chat(body)
            elif path == "/v1/embeddings":
                self._openai_embed(body)
            elif path == "/api/chat":
                self._ollama_chat(body)
            elif path in ("/api/embed", "/api/embeddings"):
                self._ollama_embed(body)
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _count(self, key: str) -> None:
        with self.lock:
            self.counter[key] += 1

    # -- OpenAI-compatible -------------------------------------------------------

    def _openai_chat(self, body: dict[str, Any]) -> None:
        self._count("chat")
        rf = body.get("response_format")
        json_format: Any = None
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                if self.reject_json_schema:
                    self._send(400, {"error": {"message": "response_format json_schema is not supported", "type": "invalid_request_error"}})
                    return
                json_format = (rf.get("json_schema") or {}).get("schema") or {}
            elif rf.get("type") == "json_object":
                json_format = "json"
        text = respond(body.get("messages", []), json_format)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 10**9
        words = text.split(" ")
        finish = "stop"
        if len(words) > max_tokens:
            text, finish = " ".join(words[:max_tokens]), "length"
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4
        completion = {
            "id": f"chatcmpl-mock-{self.counter['chat']}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(words), "total_tokens": prompt_tokens + len(words)},
        }
        if body.get("stream"):
            chunk = {**completion, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": finish}]}
            payload = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self._send(200, payload, "text/event-stream")
        else:
            self._send(200, completion)

    def _openai_embed(self, body: dict[str, Any]) -> None:
        self._count("embed")
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        dim = int(body.get("dimensions") or self.dim)
        self._send(200, {
            "object": "list",
            "model": body.get("model", "mock"),
            "data": [{"object": "embedding", "index": i, "embedding": embed(t, dim)} for i, t in enumerate(inputs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })

    # -- Ollama native -------------------------------------------------------------

    def _ollama_chat(self, body: dict[str, Any]) -> None:
        self._count("chat")
        fmt = body.get("format")
        json_format = fmt if isinstance(fmt, dict) else ("json" if fmt == "json" else None)
        text = respond(body.get("messages", []), json_format)
        num_predict = (body.get("options") or {}).get("num_predict") or 10**9
        words = text.split(" ")
        done_reason = "stop"
        if len(words) > num_predict:
            text, done_reason = " ".join(words[:num_predict]), "length"
        resp = {
            "model": body.get("model", "mock"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4,
            "eval_count": len(words),
        }
        if body.get("stream", True):
            self._send(200, (json.dumps(resp) + "\n").encode(), "application/x-ndjson")
        else:
            self._send(200, resp)

    def _ollama_embed(self, body: dict[str, Any]) -> None:
        self._count("embed")
        inputs = body.get("input", body.get("prompt", ""))
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        dim = int(body.get("dimensions") or self.dim)
        self._send(200, {"model": body.get("model", "mock"), "embeddings": [embed(t, dim) for t in inputs], "prompt_eval_count": 0})


def serve(port: int, dim: int = 64, reject_json_schema: bool = False, latency: float = 0.0) -> ThreadingHTTPServer:
    Handler.dim = dim
    Handler.reject_json_schema = reject_json_schema
    Handler.latency = latency
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--dim", type=int, default=64, help="embedding size")
    parser.add_argument("--reject-json-schema", action="store_true", help="answer 400 to json_schema response_format")
    parser.add_argument("--latency", type=float, default=0.0, help="seconds to sleep per request")
    args = parser.parse_args()
    server = serve(args.port, args.dim, args.reject_json_schema, args.latency)
    print(f"mock LLM server on http://127.0.0.1:{args.port}  (OpenAI: /v1, Ollama: /api)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
