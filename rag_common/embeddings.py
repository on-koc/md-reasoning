"""Embedding backends: Hugging Face (in-process), OpenAI-compatible APIs and Ollama.

    embedder = create_embedder(EmbeddingConfig(provider="ollama", model="bge-m3"))
    vectors = embedder.embed(["some text", "more text"])          # np.ndarray [n, dim]
    vectors = await embedder.aembed(["a query"], is_query=True)   # async variant
    embedder.dim                                                  # auto-detected size
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import EmbeddingConfig
from .llm import _PerLoop, _backoff, _fingerprint, is_transient_error, resolve_torch_device, torch_dtype_kwargs

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Provider-agnostic text embedder returning ``float32`` numpy arrays."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._dim: int | None = config.dimensions
        self._dim_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.stats = {"requests": 0, "texts": 0}

    @property
    def name(self) -> str:
        return f"{self.config.provider}/{self.config.model}"

    @property
    def dim(self) -> int:
        """Embedding dimension (probed once if not configured)."""
        if self._dim is None:
            with self._dim_lock:
                if self._dim is None:
                    probe = self.embed(["dimension probe"])
                    self._dim = int(probe.shape[1])
                    logger.info("%s: detected embedding dimension %d", self.name, self._dim)
        return self._dim

    # -- public API -------------------------------------------------------------

    def embed(self, texts: list[str], *, is_query: bool = False, instruction: str | None = None) -> np.ndarray:
        prepared = self._prepare(texts, is_query, instruction)
        if not prepared:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        bs = self.config.batch_size
        parts = [self._retry_sync(prepared[i : i + bs]) for i in range(0, len(prepared), bs)]
        return self._finalize(np.concatenate(parts, axis=0), len(prepared))

    async def aembed(self, texts: list[str], *, is_query: bool = False, instruction: str | None = None) -> np.ndarray:
        prepared = self._prepare(texts, is_query, instruction)
        if not prepared:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        bs = self.config.batch_size
        parts = await asyncio.gather(
            *(self._retry_async(prepared[i : i + bs]) for i in range(0, len(prepared), bs))
        )
        return self._finalize(np.concatenate(parts, axis=0), len(prepared))

    def close(self) -> None:  # pragma: no cover - optional override
        """Release resources."""

    # -- helpers -------------------------------------------------------------------

    def _prepare(self, texts: list[str] | str, is_query: bool, instruction: str | None) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        cfg = self.config
        if instruction:
            prefix = cfg.query_instruction_template.format(instruction=instruction)
        elif is_query:
            prefix = cfg.query_prefix
        else:
            prefix = cfg.document_prefix
        # many servers reject empty strings
        return [f"{prefix}{t}" if (t and t.strip()) else f"{prefix} " for t in texts]

    def _finalize(self, arr: np.ndarray, expected: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != expected:
            raise ValueError(f"{self.name}: expected {expected} embeddings, got array of shape {arr.shape}")
        if self.config.normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.clip(norms, 1e-12, None)
        if self._dim is not None and arr.shape[1] != self._dim:
            raise ValueError(f"{self.name}: got {arr.shape[1]}-d vectors but dimensions={self._dim}")
        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["texts"] += expected
        return arr

    def _retries(self) -> int:
        return max(0, int(self.config.max_retries))

    def _retry_sync(self, batch: list[str]) -> np.ndarray:
        attempts = self._retries() + 1
        for attempt in range(attempts):
            try:
                return self._embed_batch(batch)
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= attempts or not is_transient_error(exc):
                    raise
                delay = _backoff(attempt)
                logger.warning("%s: transient error (%s); retrying in %.1fs", self.name, exc, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")

    async def _retry_async(self, batch: list[str]) -> np.ndarray:
        attempts = self._retries() + 1
        for attempt in range(attempts):
            try:
                return await self._aembed_batch(batch)
            except Exception as exc:  # noqa: BLE001
                if attempt + 1 >= attempts or not is_transient_error(exc):
                    raise
                delay = _backoff(attempt)
                logger.warning("%s: transient error (%s); retrying in %.1fs", self.name, exc, delay)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    @abstractmethod
    def _embed_batch(self, batch: list[str]) -> np.ndarray: ...

    @abstractmethod
    async def _aembed_batch(self, batch: list[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP APIs
# ---------------------------------------------------------------------------


class OpenAIEmbedder(Embedder):
    """``POST /v1/embeddings`` on any OpenAI-compatible server (vLLM, TEI, llama.cpp, LM Studio, Ollama /v1...)."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        import openai

        self.base_url = config.base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        kwargs = dict(base_url=self.base_url, api_key=api_key, timeout=config.timeout, max_retries=config.max_retries)
        self._sync_client = openai.OpenAI(**kwargs)
        self._async = _PerLoop(lambda: (openai.AsyncOpenAI(**kwargs), asyncio.Semaphore(config.concurrency)))
        self._thread_sem = threading.BoundedSemaphore(config.concurrency)

    def _retries(self) -> int:
        return 0  # the openai client retries by itself

    def _request(self, batch: list[str]) -> dict[str, Any]:
        req: dict[str, Any] = {"model": self.config.model, "input": batch, "encoding_format": "float"}
        if self.config.send_dimensions and self.config.dimensions:
            req["dimensions"] = self.config.dimensions
        return req

    @staticmethod
    def _to_array(resp: Any) -> np.ndarray:
        data = sorted(resp.data, key=lambda d: d.index)
        return np.array([d.embedding for d in data], dtype=np.float32)

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        with self._thread_sem:
            return self._to_array(self._sync_client.embeddings.create(**self._request(batch)))

    async def _aembed_batch(self, batch: list[str]) -> np.ndarray:
        client, sem = self._async.get()
        async with sem:
            return self._to_array(await client.embeddings.create(**self._request(batch)))


# ---------------------------------------------------------------------------
# Ollama native API
# ---------------------------------------------------------------------------


class OllamaEmbedder(Embedder):
    """Ollama's native ``/api/embed`` via the ``ollama`` Python package."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        import ollama

        self.host = config.base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        api_key = config.api_key or os.environ.get("OLLAMA_API_KEY")
        kwargs: dict[str, Any] = {"host": self.host, "timeout": config.timeout}
        if api_key:
            kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
        self._sync_client = ollama.Client(**kwargs)
        self._async = _PerLoop(lambda: (ollama.AsyncClient(**kwargs), asyncio.Semaphore(config.concurrency)))
        self._thread_sem = threading.BoundedSemaphore(config.concurrency)

    def _request(self, batch: list[str]) -> dict[str, Any]:
        cfg = self.config
        req: dict[str, Any] = {"model": cfg.model, "input": batch, "truncate": cfg.truncate}
        options = dict(cfg.options)
        if cfg.num_ctx:
            options["num_ctx"] = cfg.num_ctx
        if options:
            req["options"] = options
        if cfg.keep_alive is not None:
            req["keep_alive"] = cfg.keep_alive
        if cfg.send_dimensions and cfg.dimensions:
            req["dimensions"] = cfg.dimensions
        return req

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        with self._thread_sem:
            return np.array(self._sync_client.embed(**self._request(batch)).embeddings, dtype=np.float32)

    async def _aembed_batch(self, batch: list[str]) -> np.ndarray:
        client, sem = self._async.get()
        async with sem:
            resp = await client.embed(**self._request(batch))
            return np.array(resp.embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Hugging Face (in-process)
# ---------------------------------------------------------------------------


class HuggingFaceEmbedder(Embedder):
    """Local embedding model via ``sentence-transformers`` (preferred) or plain ``transformers``."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._lock = threading.Lock()
        backend = config.hf_backend
        if backend == "auto":
            try:
                import sentence_transformers  # noqa: F401

                backend = "sentence_transformers"
            except ImportError:
                backend = "transformers"
        self.backend = backend
        device_map, device = resolve_torch_device(config.device)
        device = device or "cuda"  # embedding models are small: use one device
        logger.info("Loading Hugging Face embedding model %s (%s) on %s ...", config.model, backend, device)
        if backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            model_kwargs = {**torch_dtype_kwargs(config.dtype), **config.model_kwargs}
            if "dtype" in model_kwargs and model_kwargs["dtype"] == "auto":
                model_kwargs.pop("dtype")  # let sentence-transformers decide
            if "torch_dtype" in model_kwargs and model_kwargs["torch_dtype"] == "auto":
                model_kwargs.pop("torch_dtype")
            self.model = SentenceTransformer(
                config.model, device=device, trust_remote_code=config.trust_remote_code, model_kwargs=model_kwargs
            )
            if config.max_tokens and getattr(self.model, "max_seq_length", None):
                self.model.max_seq_length = min(self.model.max_seq_length, config.max_tokens)
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            self.model = AutoModel.from_pretrained(
                config.model,
                trust_remote_code=config.trust_remote_code,
                **torch_dtype_kwargs(config.dtype),
                **config.model_kwargs,
            ).to(device)
            self.model.eval()
            self._device = device

    def _retries(self) -> int:
        return 0

    def _encode_transformers(self, batch: list[str]) -> np.ndarray:
        torch = self._torch
        max_len = min(self.config.max_tokens, getattr(self.tokenizer, "model_max_length", 512) or 512)
        enc = self.tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if self.config.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.config.pooling == "last":
            if self.tokenizer.padding_side == "left":
                pooled = hidden[:, -1]
            else:
                lengths = enc["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        else:
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.float().cpu().numpy()

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        with self._lock:
            if self.backend == "sentence_transformers":
                return self.model.encode(
                    batch,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
            return self._encode_transformers(batch)

    async def _aembed_batch(self, batch: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._embed_batch, batch)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[Embedder]] = {
    "openai": OpenAIEmbedder,
    "ollama": OllamaEmbedder,
    "huggingface": HuggingFaceEmbedder,
}
_SHARED: dict[str, Embedder] = {}
_SHARED_LOCK = threading.Lock()


def create_embedder(config: EmbeddingConfig, *, shared: bool = True) -> Embedder:
    """Instantiate the backend for ``config.provider`` (shared per identical config)."""
    if not shared:
        return _BACKENDS[config.provider](config)
    key = _fingerprint(config)
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = _BACKENDS[config.provider](config)
        return _SHARED[key]
