"""Chat model backends: Hugging Face (in-process), OpenAI-compatible APIs and Ollama.

Every backend exposes the same small interface used by the method adapters::

    llm = create_llm(LLMConfig(provider="ollama", model="qwen2.5:14b"))
    result = llm.generate([{"role": "user", "content": "Hello"}])
    result = await llm.agenerate(messages, json_schema={...})   # async variant
    result.text, result.finish_reason   # "stop" | "length" | ...

Both the sync and the async entry points are safe to call from many threads /
tasks at once; concurrency against the model is capped by ``config.concurrency``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import logging
import os
import queue
import random
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .config import LLMConfig
from .parsing import clean_json_text, strip_thinking

logger = logging.getLogger(__name__)

Message = dict[str, Any]


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


class _PerLoop:
    """Lazily create one object per running asyncio event loop.

    Async HTTP clients and asyncio primitives are bound to the loop they were
    first used on. The methods drive our backends from several loops (worker
    threads, ``asyncio.run`` inside sync wrappers, ...), so we keep one
    instance per loop instead of sharing a single one.
    """

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._items: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def get(self) -> Any:
        loop = asyncio.get_running_loop()
        with self._lock:
            item = self._items.get(loop)
            if item is None:
                item = self._factory()
                self._items[loop] = item
            return item


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "ReadError", "WriteError",
    "TimeoutException", "NetworkError",
}


def is_transient_error(exc: BaseException) -> bool:
    """Heuristic: is ``exc`` worth retrying (timeouts, connection drops, 429/5xx)?"""
    code = _status_code(exc)
    if code is not None:
        return code in (408, 409, 425, 429) or code >= 500
    return type(exc).__name__ in _TRANSIENT_NAMES or isinstance(exc, (TimeoutError, ConnectionError))


def _backoff(attempt: int) -> float:
    return min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)


def _with_json_hint(messages: list[Message], schema: dict[str, Any] | None) -> list[Message]:
    """Append a JSON-only instruction to the last user message (for backends
    without native structured output)."""
    hint = "Respond only with a single valid JSON value and nothing else."
    if schema:
        hint += " It must conform to this JSON schema: " + json.dumps(schema, ensure_ascii=False)
    out = copy.deepcopy(messages)
    for msg in reversed(out):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            msg["content"] = f"{msg['content']}\n\n{hint}"
            return out
    out.append({"role": "user", "content": hint})
    return out


def _normalize_messages(messages: list[Message] | str) -> list[Message]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    normalized = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts -> plain text
            m["content"] = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        elif content is None:
            m["content"] = ""
        normalized.append({"role": m.get("role", "user"), "content": m["content"]})
    return normalized


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------


class ChatLLM(ABC):
    """Provider-agnostic chat model."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._stats_lock = threading.Lock()
        self.stats = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "truncated": 0}

    @property
    def name(self) -> str:
        return f"{self.config.provider}/{self.config.model}"

    # -- public API -----------------------------------------------------------

    def generate(
        self,
        messages: list[Message] | str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:
        params = self._params(max_tokens, temperature, json_mode, json_schema, stop)
        msgs = _normalize_messages(messages)
        result = self._retry_sync(lambda: self._generate(msgs, **params))
        return self._postprocess(result, params)

    async def agenerate(
        self,
        messages: list[Message] | str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
    ) -> ChatResult:
        params = self._params(max_tokens, temperature, json_mode, json_schema, stop)
        msgs = _normalize_messages(messages)
        result = await self._retry_async(lambda: self._agenerate(msgs, **params))
        return self._postprocess(result, params)

    def close(self) -> None:  # pragma: no cover - optional override
        """Release resources (HTTP clients, GPU memory)."""

    # -- helpers ----------------------------------------------------------------

    def _params(self, max_tokens, temperature, json_mode, json_schema, stop) -> dict[str, Any]:
        cfg = self.config
        return {
            "max_tokens": int(max_tokens or cfg.max_tokens),
            "temperature": cfg.temperature if temperature is None else float(temperature),
            "json_mode": bool(json_mode or json_schema),
            "json_schema": json_schema if cfg.structured_output != "json_object" else None,
            "stop": stop if stop is not None else cfg.stop,
        }

    def _postprocess(self, result: ChatResult, params: dict[str, Any]) -> ChatResult:
        text = result.text or ""
        if self.config.strip_thinking:
            text = strip_thinking(text)
        if params["json_mode"]:
            text = clean_json_text(text)
        result.text = text
        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += result.prompt_tokens or 0
            self.stats["completion_tokens"] += result.completion_tokens or 0
            if result.finish_reason == "length":
                self.stats["truncated"] += 1
        if result.finish_reason == "length":
            logger.debug("%s: completion hit max_tokens=%s", self.name, params["max_tokens"])
        return result

    def _retries(self) -> int:
        return max(0, int(self.config.max_retries))

    def _retry_sync(self, fn: Callable[[], ChatResult]) -> ChatResult:
        attempts = self._retries() + 1
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= attempts or not is_transient_error(exc):
                    raise
                delay = _backoff(attempt)
                logger.warning("%s: transient error (%s); retrying in %.1fs", self.name, exc, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    async def _retry_async(self, fn: Callable[[], Any]) -> ChatResult:
        attempts = self._retries() + 1
        for attempt in range(attempts):
            try:
                return await fn()
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= attempts or not is_transient_error(exc):
                    raise
                delay = _backoff(attempt)
                logger.warning("%s: transient error (%s); retrying in %.1fs", self.name, exc, delay)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    # -- to implement -------------------------------------------------------------

    @abstractmethod
    def _generate(self, messages: list[Message], **params: Any) -> ChatResult: ...

    @abstractmethod
    async def _agenerate(self, messages: list[Message], **params: Any) -> ChatResult: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP APIs
# ---------------------------------------------------------------------------


class OpenAIChat(ChatLLM):
    """Any server implementing ``POST /v1/chat/completions`` (vLLM, SGLang,
    llama.cpp, LM Studio, TGI, LocalAI, Ollama's /v1, OpenAI itself, ...)."""

    _MODES = ("json_schema", "json_object", "none")

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import openai

        self._openai = openai
        self.base_url = config.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        client_kwargs = dict(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,  # the openai client retries 408/409/429/5xx itself
        )
        self._sync_client = openai.OpenAI(**client_kwargs)
        self._async = _PerLoop(
            lambda: (openai.AsyncOpenAI(**client_kwargs), asyncio.Semaphore(config.concurrency))
        )
        self._thread_sem = threading.BoundedSemaphore(config.concurrency)
        # structured-output capability, degraded at runtime if the server rejects it
        self._mode_floor = 0 if config.structured_output in ("auto", "json_schema") else (
            1 if config.structured_output == "json_object" else 2
        )
        self._mode_lock = threading.Lock()

    def _retries(self) -> int:
        return 0  # handled by the openai client

    def _mode(self, params: dict[str, Any]) -> str:
        if not params["json_mode"]:
            return "none"
        wanted = 0 if params["json_schema"] else 1
        return self._MODES[max(wanted, self._mode_floor)]

    def _request(self, messages: list[Message], params: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        mode = self._mode(params)
        if params["json_mode"] and mode == "none":
            messages = _with_json_hint(messages, params["json_schema"])
        req: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
        }
        if cfg.top_p is not None:
            req["top_p"] = cfg.top_p
        if cfg.seed is not None:
            req["seed"] = cfg.seed
        if params["stop"]:
            req["stop"] = params["stop"]
        if mode == "json_schema":
            req["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": params["json_schema"], "strict": False},
            }
        elif mode == "json_object":
            req["response_format"] = {"type": "json_object"}
        if cfg.extra_body:
            req["extra_body"] = copy.deepcopy(cfg.extra_body)
        if cfg.extra_headers:
            req["extra_headers"] = dict(cfg.extra_headers)
        return req

    def _degrade(self, exc: BaseException, req: dict[str, Any]) -> bool:
        """Lower the structured-output mode after the server rejected ``response_format``."""
        if "response_format" not in req or _status_code(exc) not in (400, 404, 415, 422, 500, 501):
            return False
        msg = str(exc).lower()
        if not any(s in msg for s in ("response_format", "json", "schema", "grammar", "guided", "format")):
            return False
        with self._mode_lock:
            current = self._MODES.index(
                "json_schema" if req["response_format"]["type"] == "json_schema" else "json_object"
            )
            if self._mode_floor <= current:
                self._mode_floor = current + 1
                logger.warning(
                    "%s rejected response_format=%s; falling back to %s",
                    self.base_url, self._MODES[current], self._MODES[self._mode_floor],
                )
        return True

    @staticmethod
    def _to_result(resp: Any) -> ChatResult:
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def _generate(self, messages: list[Message], **params: Any) -> ChatResult:
        with self._thread_sem:
            for _ in range(len(self._MODES)):
                req = self._request(messages, params)
                try:
                    return self._to_result(self._sync_client.chat.completions.create(**req))
                except self._openai.APIStatusError as exc:
                    if not self._degrade(exc, req):
                        raise
        raise RuntimeError("structured output negotiation failed")

    async def _agenerate(self, messages: list[Message], **params: Any) -> ChatResult:
        client, sem = self._async.get()
        async with sem:
            for _ in range(len(self._MODES)):
                req = self._request(messages, params)
                try:
                    return self._to_result(await client.chat.completions.create(**req))
                except self._openai.APIStatusError as exc:
                    if not self._degrade(exc, req):
                        raise
        raise RuntimeError("structured output negotiation failed")

    def close(self) -> None:
        self._sync_client.close()


# ---------------------------------------------------------------------------
# Ollama native API
# ---------------------------------------------------------------------------


class OllamaChat(ChatLLM):
    """Ollama's native ``/api/chat`` via the ``ollama`` Python package."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import ollama

        self.host = config.base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        api_key = config.api_key or os.environ.get("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        client_kwargs: dict[str, Any] = {"host": self.host, "timeout": config.timeout}
        if headers:
            client_kwargs["headers"] = headers
        self._sync_client = ollama.Client(**client_kwargs)
        self._async = _PerLoop(
            lambda: (ollama.AsyncClient(**client_kwargs), asyncio.Semaphore(config.concurrency))
        )
        self._thread_sem = threading.BoundedSemaphore(config.concurrency)

    def _request(self, messages: list[Message], params: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        options: dict[str, Any] = {"temperature": params["temperature"], "num_predict": params["max_tokens"]}
        if cfg.num_ctx:
            options["num_ctx"] = cfg.num_ctx
        if cfg.top_p is not None:
            options["top_p"] = cfg.top_p
        if cfg.seed is not None:
            options["seed"] = cfg.seed
        if params["stop"]:
            options["stop"] = params["stop"]
        options.update(cfg.options)
        req: dict[str, Any] = {"model": cfg.model, "messages": messages, "options": options}
        if params["json_mode"] and cfg.structured_output != "none":
            req["format"] = params["json_schema"] or "json"
        elif params["json_mode"]:
            req["messages"] = _with_json_hint(messages, params["json_schema"])
        if cfg.keep_alive is not None:
            req["keep_alive"] = cfg.keep_alive
        if cfg.think is not None:
            req["think"] = cfg.think
        return req

    @staticmethod
    def _to_result(resp: Any) -> ChatResult:
        return ChatResult(
            text=resp.message.content or "",
            finish_reason=getattr(resp, "done_reason", None),
            prompt_tokens=getattr(resp, "prompt_eval_count", None),
            completion_tokens=getattr(resp, "eval_count", None),
        )

    def _generate(self, messages: list[Message], **params: Any) -> ChatResult:
        with self._thread_sem:
            return self._to_result(self._sync_client.chat(**self._request(messages, params)))

    async def _agenerate(self, messages: list[Message], **params: Any) -> ChatResult:
        client, sem = self._async.get()
        async with sem:
            return self._to_result(await client.chat(**self._request(messages, params)))


# ---------------------------------------------------------------------------
# Hugging Face transformers (in-process)
# ---------------------------------------------------------------------------


def resolve_torch_device(device: str | None) -> tuple[str | None, str | None]:
    """Return ``(device_map, device)``: exactly one of them is set."""
    import torch

    if device in (None, "", "auto"):
        if torch.cuda.is_available():
            try:
                import accelerate  # noqa: F401

                return "auto", None
            except ImportError:
                return None, "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return None, "mps"
        return None, "cpu"
    return None, device


def torch_dtype_kwargs(dtype: str | None) -> dict[str, Any]:
    """``from_pretrained`` dtype kwarg (``dtype`` in transformers>=4.56, ``torch_dtype`` before)."""
    import torch
    import transformers
    from packaging import version

    aliases = {"bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32", "float": "float32"}
    if dtype in (None, "", "auto"):
        value: Any = "auto"
    else:
        name = aliases.get(str(dtype).lower(), str(dtype).lower())
        if not isinstance(getattr(torch, name, None), torch.dtype):
            raise ValueError(f"unknown dtype {dtype!r}; use auto, bfloat16, float16 or float32")
        value = getattr(torch, name)
    key = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"
    return {key: value}


class _Request:
    __slots__ = ("prompt", "params", "future")

    def __init__(self, prompt: str, params: dict[str, Any]):
        self.prompt = prompt
        self.params = params
        self.future: concurrent.futures.Future = concurrent.futures.Future()


class _GenerationBatcher:
    """Collects concurrent requests and runs them through one batched ``generate``."""

    def __init__(self, run_batch: Callable[[list[str], dict[str, Any]], list[ChatResult]], max_batch: int, wait_ms: int):
        self._run_batch = run_batch
        self._max_batch = max_batch
        self._wait = wait_ms / 1000.0
        self._queue: queue.Queue[_Request] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="hf-generate-batcher", daemon=True)
        self._thread.start()

    def submit(self, prompt: str, params: dict[str, Any]) -> concurrent.futures.Future:
        req = _Request(prompt, params)
        self._queue.put(req)
        return req.future

    def _loop(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + self._wait
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            groups: dict[str, list[_Request]] = {}
            for req in batch:
                groups.setdefault(json.dumps(req.params, sort_keys=True, default=str), []).append(req)
            for reqs in groups.values():
                try:
                    results = self._run_batch([r.prompt for r in reqs], reqs[0].params)
                except BaseException as exc:  # noqa: BLE001 - propagate to callers
                    for r in reqs:
                        r.future.set_exception(exc)
                else:
                    for r, res in zip(reqs, results):
                        r.future.set_result(res)


class HuggingFaceChat(ChatLLM):
    """Runs a causal LM locally with ``transformers`` (``AutoModelForCausalLM``).

    There is no constrained decoding here: JSON requests add an instruction
    (incl. the schema) to the prompt and the output is repaired afterwards.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        logger.info("Loading Hugging Face model %s ...", config.model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation with decoder-only models

        device_map, device = resolve_torch_device(config.device)
        load_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code, **torch_dtype_kwargs(config.dtype)}
        if config.quantization:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                if config.quantization == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
            device_map = device_map or "auto"
        if device_map:
            load_kwargs["device_map"] = device_map
        load_kwargs.update(config.model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
        if not device_map and device:
            self.model.to(device)
        self.model.eval()

        gen_cfg = getattr(self.model, "generation_config", None)
        eos = getattr(gen_cfg, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self._eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        self._has_template = bool(getattr(self.tokenizer, "chat_template", None))
        if not self._has_template:
            logger.warning("%s has no chat template; using a plain-text prompt format", config.model)

        self._lock = threading.Lock()
        self._batcher = (
            _GenerationBatcher(self._run_batch, config.batch_size, config.batch_wait_ms)
            if config.batch_size > 1
            else None
        )

    def _retries(self) -> int:
        return 0

    # -- prompt rendering --------------------------------------------------------

    def render(self, messages: list[Message]) -> str:
        if not self._has_template:
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            return "\n\n".join(parts) + "\n\nAssistant:"
        kwargs = dict(tokenize=False, add_generation_prompt=True, **self.config.chat_template_kwargs)
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:  # noqa: BLE001 - e.g. templates that reject the system role
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            rest = [dict(m) for m in messages if m["role"] != "system"]
            if system and rest and rest[0]["role"] == "user":
                rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
            elif system:
                rest.insert(0, {"role": "user", "content": system})
            return self.tokenizer.apply_chat_template(rest, **kwargs)

    # -- generation ----------------------------------------------------------------

    def _run_batch(self, prompts: list[str], params: dict[str, Any]) -> list[ChatResult]:
        torch = self._torch
        tok = self.tokenizer
        inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=not self._has_template)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items() if k != "token_type_ids"}
        gen_kwargs: dict[str, Any] = {"max_new_tokens": params["max_tokens"], "pad_token_id": tok.pad_token_id}
        if params["temperature"] and params["temperature"] > 0:
            gen_kwargs.update(do_sample=True, temperature=params["temperature"])
            if self.config.top_p is not None:
                gen_kwargs["top_p"] = self.config.top_p
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        if params["stop"]:
            gen_kwargs.update(stop_strings=list(params["stop"]), tokenizer=tok)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        with self._lock, torch.inference_mode():
            output = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        prompt_tokens = inputs["attention_mask"].sum(dim=1).tolist()
        results = []
        for row, n_prompt in zip(output[:, prompt_len:].tolist(), prompt_tokens):
            cut = next((i for i, t in enumerate(row) if t in self._eos_ids), None)
            ids = row if cut is None else row[:cut]
            if cut is None and tok.pad_token_id is not None and tok.pad_token_id not in self._eos_ids:
                while ids and ids[-1] == tok.pad_token_id:
                    ids = ids[:-1]
            finished = cut is not None or len(row) < params["max_tokens"]
            results.append(
                ChatResult(
                    text=tok.decode(ids, skip_special_tokens=True),
                    finish_reason="stop" if finished else "length",
                    prompt_tokens=int(n_prompt),
                    completion_tokens=len(ids),
                )
            )
        return results

    def _prepare(self, messages: list[Message], params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if params["json_mode"]:
            messages = _with_json_hint(messages, params["json_schema"])
        gen_params = {k: params[k] for k in ("max_tokens", "temperature", "stop")}
        return self.render(messages), gen_params

    def _generate(self, messages: list[Message], **params: Any) -> ChatResult:
        prompt, gen_params = self._prepare(messages, params)
        if self._batcher is not None:
            return self._batcher.submit(prompt, gen_params).result()
        return self._run_batch([prompt], gen_params)[0]

    async def _agenerate(self, messages: list[Message], **params: Any) -> ChatResult:
        prompt, gen_params = self._prepare(messages, params)
        if self._batcher is not None:
            return await asyncio.wrap_future(self._batcher.submit(prompt, gen_params))
        return await asyncio.to_thread(lambda: self._run_batch([prompt], gen_params)[0])


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[ChatLLM]] = {
    "openai": OpenAIChat,
    "ollama": OllamaChat,
    "huggingface": HuggingFaceChat,
}
_SHARED: dict[str, ChatLLM] = {}
_SHARED_LOCK = threading.Lock()


def _fingerprint(config: Any) -> str:
    import dataclasses

    return json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)


def create_llm(config: LLMConfig, *, shared: bool = True) -> ChatLLM:
    """Instantiate the backend for ``config.provider``.

    With ``shared=True`` identical configs return the same instance, so a
    Hugging Face model is only loaded once per process even if a framework
    asks for several model objects.
    """
    if not shared:
        return _BACKENDS[config.provider](config)
    key = _fingerprint(config)
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = _BACKENDS[config.provider](config)
        return _SHARED[key]
