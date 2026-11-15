"""Plug the rag_common backends (Hugging Face / OpenAI-compatible / Ollama) into
Microsoft GraphRAG 3.x.

GraphRAG 3 talks to models through ``graphrag-llm``, which lets you register
custom ``LLMCompletion`` / ``LLMEmbedding`` implementations under a model
``type``. After :func:`register` has run, a ``settings.yaml`` entry like::

    completion_models:
      default_completion_model:
        type: local                 # <- the type registered here
        model_provider: ollama      # huggingface | openai | ollama
        model: qwen2.5:14b-instruct
        api_base: http://localhost:11434
        num_ctx: 32768              # any rag_common LLMConfig field
        retry: {type: exponential_backoff, max_retries: 5}

is served by our backend, while GraphRAG's own middleware (cache, retries,
rate limiting, metrics) keeps working unchanged.

Embeddings work the same way under ``embedding_models`` (any
``EmbeddingConfig`` field may be added to the entry).
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Unpack, get_args, get_origin

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graphrag_llm.completion import LLMCompletion, register_completion  # noqa: E402
from graphrag_llm.config import TokenizerConfig, TokenizerType  # noqa: E402
from graphrag_llm.embedding import LLMEmbedding, register_embedding  # noqa: E402
from graphrag_llm.middleware import with_middleware_pipeline  # noqa: E402
from graphrag_llm.tokenizer import create_tokenizer  # noqa: E402
from graphrag_llm.types import (  # noqa: E402
    LLMChoice,
    LLMChoiceChunk,
    LLMChoiceDelta,
    LLMCompletionChunk,
    LLMCompletionMessage,
    LLMCompletionResponse,
    LLMCompletionUsage,
    LLMEmbedding as LLMEmbeddingItem,
    LLMEmbeddingResponse,
    LLMEmbeddingUsage,
)
from graphrag_llm.utils import structure_completion_response  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from rag_common.config import ConfigError, EmbeddingConfig, LLMConfig  # noqa: E402
from rag_common.embeddings import create_embedder  # noqa: E402
from rag_common.llm import ChatResult, create_llm  # noqa: E402
from rag_common.parsing import extract_json, json_schema_of  # noqa: E402

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import LLMCompletionArgs, LLMEmbeddingArgs, ResponseFormat

logger = logging.getLogger(__name__)

LOCAL_TYPE = "local"
"""Value of ``type:`` in ``completion_models`` / ``embedding_models`` entries."""

DEFAULT_ENCODING = "o200k_base"

# extras that GraphRAG itself may put on a ModelConfig and that we must ignore
_IGNORED_EXTRAS = {"encoding_model", "failure_rate_for_testing", "failure_rate_for_testing_exception_type",
                   "failure_rate_for_testing_exception_args"}


# ---------------------------------------------------------------------------
# ModelConfig  ->  rag_common configs
# ---------------------------------------------------------------------------


def _extras(model_config: ModelConfig, cls: type) -> dict[str, Any]:
    extra = dict(model_config.model_extra or {})
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(extra) - valid - _IGNORED_EXTRAS)
    if unknown:
        raise ConfigError(
            f"Unknown option(s) {unknown} on GraphRAG model '{model_config.model}' (type: {LOCAL_TYPE}). "
            f"Valid options: {sorted(valid)}"
        )
    return {k: v for k, v in extra.items() if k in valid}


def llm_config_from_model_config(model_config: ModelConfig) -> LLMConfig:
    data: dict[str, Any] = {"provider": model_config.model_provider, "model": model_config.model}
    if model_config.api_base:
        data["base_url"] = model_config.api_base
    if model_config.api_key:
        data["api_key"] = model_config.api_key
    data.update(_extras(model_config, LLMConfig))
    return LLMConfig.from_dict(data)


def embedding_config_from_model_config(model_config: ModelConfig) -> EmbeddingConfig:
    data: dict[str, Any] = {"provider": model_config.model_provider, "model": model_config.model}
    if model_config.api_base:
        data["base_url"] = model_config.api_base
    if model_config.api_key:
        data["api_key"] = model_config.api_key
    data.update(_extras(model_config, EmbeddingConfig))
    return EmbeddingConfig.from_dict(data)


def _tokenizer_for(model_config: ModelConfig) -> Tokenizer:
    """A tiktoken tokenizer for GraphRAG's token budgeting.

    graphrag-llm defaults to LiteLLM's tokenizer selection, which may try to
    download Hugging Face tokenizers for some model names. Token counts are
    only used for budgeting, so a local tiktoken encoding is good enough.
    """
    encoding = (model_config.model_extra or {}).get("encoding_model") or DEFAULT_ENCODING
    return create_tokenizer(TokenizerConfig(type=TokenizerType.Tiktoken, encoding_name=encoding))


def _scoped_cache_key_creator(creator: CacheKeyCreator, scope: str) -> CacheKeyCreator:
    """Include the backend/model in the cache key so switching models never
    returns another model's cached answers."""

    def _create(input_args: dict[str, Any]) -> str:
        return creator({**input_args, "_local_model": scope})

    return _create


# ---------------------------------------------------------------------------
# response helpers
# ---------------------------------------------------------------------------


def _finish_reason(result: ChatResult) -> str:
    return "length" if result.finish_reason == "length" else "stop"


def _completion_response(content: str, result: ChatResult, model: str) -> LLMCompletionResponse:
    prompt = result.prompt_tokens or 0
    completion = result.completion_tokens or 0
    return LLMCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            LLMChoice(
                index=0,
                message=LLMCompletionMessage(role="assistant", content=content),
                finish_reason=_finish_reason(result),
            )
        ],
        usage=LLMCompletionUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
    )


def _chunks(content: str, result: ChatResult, model: str) -> list[LLMCompletionChunk]:
    """Pseudo-streaming: the full answer as one delta, then a final chunk."""
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

    def _chunk(delta: LLMChoiceDelta, finish: str | None) -> LLMCompletionChunk:
        return LLMCompletionChunk(
            id=cid,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[LLMChoiceChunk(index=0, delta=delta, finish_reason=finish)],
        )

    return [
        _chunk(LLMChoiceDelta(role="assistant", content=content), None),
        _chunk(LLMChoiceDelta(), _finish_reason(result)),
    ]


def _empty_value(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if annotation is str:
        return ""
    if annotation in (int, float):
        return annotation(0)
    if annotation is bool:
        return False
    if annotation is list or origin is list:
        return []
    if annotation is dict or origin is dict:
        return {}
    if origin is not None and type(None) in get_args(annotation):
        return None
    raise KeyError(annotation)


def validate_structured(text: str, response_format: type[BaseModel]) -> BaseModel:
    """Parse model output into ``response_format``.

    Local models often wrap JSON in prose / code fences or drop a field; we
    extract + repair the JSON and fill trivially-missing scalar/list fields
    before giving up.
    """
    obj = extract_json(text)
    try:
        return response_format.model_validate(obj)
    except ValidationError as err:
        if not isinstance(obj, dict):
            raise
        patched = dict(obj)
        for name, field in response_format.model_fields.items():
            if name not in patched and field.is_required():
                try:
                    patched[name] = _empty_value(field.annotation)
                except KeyError:
                    raise err from None
        return response_format.model_validate(patched)


# ---------------------------------------------------------------------------
# completion
# ---------------------------------------------------------------------------


class LocalCompletion(LLMCompletion):
    """``LLMCompletion`` backed by a rag_common :class:`~rag_common.llm.ChatLLM`."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: ModelConfig,
        tokenizer: Tokenizer,
        metrics_store: MetricsStore,
        metrics_processor: MetricsProcessor | None = None,
        rate_limiter: RateLimiter | None = None,
        retrier: Retry | None = None,
        cache: Cache | None = None,
        cache_key_creator: CacheKeyCreator,
        **kwargs: Any,
    ) -> None:
        self._model_config = model_config
        self._llm = create_llm(llm_config_from_model_config(model_config))
        self._tokenizer = _tokenizer_for(model_config)
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._track_metrics = metrics_processor is not None
        self._completion, self._completion_async = with_middleware_pipeline(
            model_config=model_config,
            model_fn=self._base_completion,
            async_model_fn=self._base_completion_async,
            request_type="chat",
            cache=cache,
            cache_key_creator=_scoped_cache_key_creator(cache_key_creator, self._llm.name),
            tokenizer=self._tokenizer,
            metrics_processor=metrics_processor,
            rate_limiter=rate_limiter,
            retrier=retrier,
        )

    # -- argument mapping -----------------------------------------------------------

    def _gen_kwargs(self, kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        call = {**(self._model_config.call_args or {}), **kwargs}
        out: dict[str, Any] = {}
        max_tokens = call.get("max_completion_tokens") or call.get("max_tokens")
        if max_tokens:
            out["max_tokens"] = int(max_tokens)
        if call.get("temperature") is not None:
            out["temperature"] = float(call["temperature"])
        if call.get("stop"):
            out["stop"] = call["stop"] if isinstance(call["stop"], list) else [call["stop"]]
        response_format = call.get("response_format")
        schema = json_schema_of(response_format)
        if schema:
            out["json_schema"] = schema
        elif response_format is not None or call.get("response_format_json_object"):
            out["json_mode"] = True
        return out, response_format

    @staticmethod
    def _structured_model(response_format: Any) -> type[BaseModel] | None:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format
        return None

    def _check(self, result: ChatResult, response_format: Any) -> str:
        model = self._structured_model(response_format)
        if model is None:
            return result.text
        # validate here (inside the cache/retry middleware) so that invalid
        # JSON is retried instead of being cached
        return validate_structured(result.text, model).model_dump_json()

    # -- base functions wrapped by GraphRAG's middleware ------------------------

    def _base_completion(self, **kwargs: Any) -> LLMCompletionResponse | Iterator[LLMCompletionChunk]:
        kwargs.pop("metrics", None)
        messages = kwargs.pop("messages")
        stream = bool(kwargs.pop("stream", False))
        kwargs.pop("stream_options", None)
        gen, response_format = self._gen_kwargs(kwargs)
        result = self._llm.generate(messages, **gen)
        try:
            content = self._check(result, response_format)
        except (ValueError, ValidationError) as exc:
            logger.warning("structured output invalid (%s); retrying once with sampling", exc)
            result = self._llm.generate(messages, **{**gen, "temperature": max(0.7, gen.get("temperature", 0.0))})
            content = self._check(result, response_format)
        if stream:
            return iter(_chunks(content, result, self._llm.name))
        return _completion_response(content, result, self._llm.name)

    async def _base_completion_async(
        self, **kwargs: Any
    ) -> LLMCompletionResponse | AsyncIterator[LLMCompletionChunk]:
        kwargs.pop("metrics", None)
        messages = kwargs.pop("messages")
        stream = bool(kwargs.pop("stream", False))
        kwargs.pop("stream_options", None)
        gen, response_format = self._gen_kwargs(kwargs)
        result = await self._llm.agenerate(messages, **gen)
        try:
            content = self._check(result, response_format)
        except (ValueError, ValidationError) as exc:
            logger.warning("structured output invalid (%s); retrying once with sampling", exc)
            result = await self._llm.agenerate(messages, **{**gen, "temperature": max(0.7, gen.get("temperature", 0.0))})
            content = self._check(result, response_format)
        if stream:
            chunks = _chunks(content, result, self._llm.name)

            async def _aiter() -> AsyncIterator[LLMCompletionChunk]:
                for chunk in chunks:
                    yield chunk

            return _aiter()
        return _completion_response(content, result, self._llm.name)

    # -- LLMCompletion API ---------------------------------------------------------

    def _prepare(self, kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
        messages = kwargs.pop("messages")
        response_format = kwargs.pop("response_format", None)
        if response_format is not None and kwargs.get("stream"):
            raise ValueError("response_format is not supported for streaming completions.")
        metrics = (kwargs.pop("metrics", None) or {}) if self._track_metrics else None
        kwargs.pop("metrics", None)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return messages, response_format, metrics

    def completion(
        self, /, **kwargs: Unpack[LLMCompletionArgs[ResponseFormat]]
    ) -> LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]:
        messages, response_format, metrics = self._prepare(kwargs)  # type: ignore[arg-type]
        try:
            response = self._completion(messages=messages, metrics=metrics, response_format=response_format, **kwargs)
            if response_format is not None:
                response.formatted_response = structure_completion_response(response.content, response_format)
            return response
        finally:
            if metrics is not None:
                self._metrics_store.update_metrics(metrics=metrics)

    async def completion_async(
        self, /, **kwargs: Unpack[LLMCompletionArgs[ResponseFormat]]
    ) -> LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]:
        messages, response_format, metrics = self._prepare(kwargs)  # type: ignore[arg-type]
        try:
            response = await self._completion_async(
                messages=messages, metrics=metrics, response_format=response_format, **kwargs
            )
            if response_format is not None:
                response.formatted_response = structure_completion_response(response.content, response_format)
            return response
        finally:
            if metrics is not None:
                self._metrics_store.update_metrics(metrics=metrics)

    @property
    def metrics_store(self) -> MetricsStore:
        return self._metrics_store

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def backend(self):
        """The underlying rag_common ChatLLM (for usage statistics)."""
        return self._llm


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


class LocalEmbedding(LLMEmbedding):
    """``LLMEmbedding`` backed by a rag_common :class:`~rag_common.embeddings.Embedder`."""

    def __init__(
        self,
        *,
        model_id: str,
        model_config: ModelConfig,
        tokenizer: Tokenizer,
        metrics_store: MetricsStore,
        metrics_processor: MetricsProcessor | None = None,
        rate_limiter: RateLimiter | None = None,
        retrier: Retry | None = None,
        cache: Cache | None = None,
        cache_key_creator: CacheKeyCreator,
        **kwargs: Any,
    ) -> None:
        self._model_config = model_config
        self._embedder = create_embedder(embedding_config_from_model_config(model_config))
        self._tokenizer = _tokenizer_for(model_config)
        self._metrics_store = metrics_store
        self._track_metrics = metrics_processor is not None
        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=model_config,
            model_fn=self._base_embedding,
            async_model_fn=self._base_embedding_async,
            request_type="embedding",
            cache=cache,
            cache_key_creator=_scoped_cache_key_creator(cache_key_creator, self._embedder.name),
            tokenizer=self._tokenizer,
            metrics_processor=metrics_processor,
            rate_limiter=rate_limiter,
            retrier=retrier,
        )

    def _response(self, vectors: Any) -> LLMEmbeddingResponse:
        return LLMEmbeddingResponse(
            object="list",
            data=[
                LLMEmbeddingItem(object="embedding", embedding=[float(x) for x in vec], index=i)
                for i, vec in enumerate(vectors)
            ],
            model=self._embedder.name,
            usage=LLMEmbeddingUsage(prompt_tokens=0, total_tokens=0),
        )

    def _base_embedding(self, **kwargs: Any) -> LLMEmbeddingResponse:
        return self._response(self._embedder.embed(list(kwargs["input"])))

    async def _base_embedding_async(self, **kwargs: Any) -> LLMEmbeddingResponse:
        return self._response(await self._embedder.aembed(list(kwargs["input"])))

    def embedding(self, /, **kwargs: Unpack[LLMEmbeddingArgs]) -> LLMEmbeddingResponse:
        metrics = (kwargs.pop("metrics", None) or {}) if self._track_metrics else None  # type: ignore[misc]
        kwargs.pop("metrics", None)  # type: ignore[misc]
        try:
            return self._embedding(metrics=metrics, **kwargs)
        finally:
            if metrics is not None:
                self._metrics_store.update_metrics(metrics=metrics)

    async def embedding_async(self, /, **kwargs: Unpack[LLMEmbeddingArgs]) -> LLMEmbeddingResponse:
        metrics = (kwargs.pop("metrics", None) or {}) if self._track_metrics else None  # type: ignore[misc]
        kwargs.pop("metrics", None)  # type: ignore[misc]
        try:
            return await self._embedding_async(metrics=metrics, **kwargs)
        finally:
            if metrics is not None:
                self._metrics_store.update_metrics(metrics=metrics)

    @property
    def metrics_store(self) -> MetricsStore:
        return self._metrics_store

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def backend(self):
        return self._embedder


_REGISTERED = False


def register() -> None:
    """Register the ``local`` model type with graphrag-llm (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_completion(LOCAL_TYPE, LocalCompletion)
    register_embedding(LOCAL_TYPE, LocalEmbedding)
    _REGISTERED = True
