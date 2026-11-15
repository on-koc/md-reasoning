"""Turn rag_common backends into LightRAG's ``llm_model_func`` and ``EmbeddingFunc``.

LightRAG calls ``llm_model_func(prompt, system_prompt=..., history_messages=...,
**kwargs)`` and expects a string. Structured output is requested with an
OpenAI-style ``response_format`` (or the legacy ``keyword_extraction`` /
``entity_extraction`` flags); we translate those into JSON / JSON-schema mode of
the selected backend (Ollama ``format``, OpenAI ``response_format``, or a JSON
instruction for in-process Hugging Face models).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rag_common.documents import ensure_tiktoken_cache  # noqa: E402
from rag_common.embeddings import Embedder  # noqa: E402
from rag_common.llm import ChatLLM  # noqa: E402
from rag_common.parsing import json_schema_of  # noqa: E402

logger = logging.getLogger(__name__)

# LightRAG tokenizes with tiktoken; use LiteLLM's bundled encodings when offline
ensure_tiktoken_cache()

# kwargs LightRAG (or its bindings) may pass that have no meaning for our backends
_IGNORED_KWARGS = {
    "hashing_kv", "token_tracker", "enable_cot", "image_inputs", "timeout", "api_key",
    "host", "base_url", "options", "stream_options", "n",
}


def _mark_truncated(text: str) -> str:
    """Flag output that hit the token limit so LightRAG does not cache it."""
    try:
        from lightrag.utils import TruncatedResponse
    except ImportError:  # older LightRAG versions
        return text
    return TruncatedResponse(text)


def make_llm_model_func(llm: ChatLLM) -> Callable[..., Any]:
    """Build an async ``llm_model_func`` for ``LightRAG(llm_model_func=...)``."""

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        keyword_extraction: bool = False,
        entity_extraction: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        response_format = kwargs.pop("response_format", None)
        stream = bool(kwargs.pop("stream", False))
        max_tokens = kwargs.pop("max_tokens", None) or kwargs.pop("max_completion_tokens", None)
        temperature = kwargs.pop("temperature", None)
        for key in list(kwargs):
            if key in _IGNORED_KWARGS:
                kwargs.pop(key)
        if kwargs:
            logger.debug("ignoring unsupported llm kwargs: %s", sorted(kwargs))

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})

        schema = json_schema_of(response_format)
        json_mode = bool(schema) or keyword_extraction or entity_extraction or response_format is not None
        result = await llm.agenerate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            json_schema=schema,
        )
        text = _mark_truncated(result.text) if result.finish_reason == "length" else result.text
        if stream:

            async def _single_chunk() -> AsyncIterator[str]:
                yield str(text)

            return _single_chunk()
        return text

    return llm_model_func


def make_embedding_func(embedder: Embedder):
    """Build a LightRAG ``EmbeddingFunc`` (dimension auto-detected)."""
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], context: str = "document", **_: Any) -> np.ndarray:
        return await embedder.aembed(list(texts), is_query=(context == "query"))

    kwargs: dict[str, Any] = {
        "embedding_dim": embedder.dim,
        "max_token_size": embedder.config.max_tokens,
        "func": embed,
    }
    fields = {f.name for f in dataclasses.fields(EmbeddingFunc)}
    if "model_name" in fields:
        kwargs["model_name"] = embedder.config.model
    if "supports_asymmetric" in fields:
        kwargs["supports_asymmetric"] = True  # LightRAG passes context="query" for search queries
    return EmbeddingFunc(**kwargs)
