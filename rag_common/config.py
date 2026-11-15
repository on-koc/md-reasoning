"""Unified model + input configuration shared by every indexing script.

All three methods (MS GraphRAG, LightRAG, HippoRAG 2) read the same ``llm:``,
``embedding:`` and ``input:`` sections, so switching a local model or provider
works the same way everywhere.  Each method additionally owns its own section
(``graphrag:``, ``lightrag:``, ``hipporag:``) that is passed through untouched.

Supported providers
-------------------
``huggingface``  Load the model in-process with ``transformers`` /
                 ``sentence-transformers``.
``openai``       Any OpenAI-compatible HTTP API (vLLM, SGLang, llama.cpp server,
                 LM Studio, TGI, LocalAI, Ollama's ``/v1`` endpoint, ...) called
                 through the official ``openai`` Python package.
``ollama``       Ollama's native API through the ``ollama`` Python package.
"""

from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROVIDERS = ("huggingface", "openai", "ollama")

_PROVIDER_ALIASES = {
    # in-process Hugging Face models
    "huggingface": "huggingface",
    "hf": "huggingface",
    "transformers": "huggingface",
    "sentence_transformers": "huggingface",
    "sentence-transformers": "huggingface",
    "local": "huggingface",
    # OpenAI-compatible servers
    "openai": "openai",
    "openai_compatible": "openai",
    "openai-compatible": "openai",
    "vllm": "openai",
    "sglang": "openai",
    "llamacpp": "openai",
    "llama.cpp": "openai",
    "llama_cpp": "openai",
    "lmstudio": "openai",
    "lm_studio": "openai",
    "tgi": "openai",
    "localai": "openai",
    # Ollama native API
    "ollama": "ollama",
}


def normalize_provider(name: str) -> str:
    """Map a provider name or alias (e.g. ``vllm``, ``hf``) to a canonical provider."""
    key = str(name or "").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(
            f"Unknown provider {name!r}. Use one of {', '.join(PROVIDERS)} "
            f"(aliases: {', '.join(sorted(k for k in _PROVIDER_ALIASES if k not in PROVIDERS))})."
        )
    return _PROVIDER_ALIASES[key]


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


def _from_dict(cls, data: dict[str, Any] | None, section: str):
    """Build dataclass ``cls`` from ``data`` and reject unknown keys (catches typos)."""
    data = dict(data or {})
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{section}' section: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(valid))}"
        )
    return cls(**data)


@dataclass
class LLMConfig:
    """Chat / completion model settings."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b-instruct"
    base_url: str | None = None
    """OpenAI: e.g. ``http://localhost:8000/v1``. Ollama: e.g. ``http://localhost:11434``."""
    api_key: str | None = None
    """Most local OpenAI-compatible servers accept any key. Falls back to OPENAI_API_KEY / OLLAMA_API_KEY."""

    # --- generation -----------------------------------------------------
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 4096
    """Maximum number of *new* tokens per completion."""
    seed: int | None = None
    stop: list[str] | None = None

    # --- runtime ----------------------------------------------------------
    timeout: float = 600.0
    max_retries: int = 3
    concurrency: int = 4
    """Maximum number of requests in flight against the model."""
    strip_thinking: bool = True
    """Remove ``<think>...</think>`` blocks emitted by reasoning models."""
    structured_output: str = "auto"
    """How JSON output is requested: ``auto`` | ``json_schema`` | ``json_object`` | ``none``.
    ``auto`` tries JSON-schema constrained decoding first and degrades gracefully."""

    # --- OpenAI-compatible only -----------------------------------------
    extra_body: dict[str, Any] = field(default_factory=dict)
    """Extra JSON fields for the request body, e.g. vLLM
    ``{"chat_template_kwargs": {"enable_thinking": false}}``."""
    extra_headers: dict[str, str] = field(default_factory=dict)

    # --- Ollama only ------------------------------------------------------
    num_ctx: int | None = 32768
    """Context window. Ollama's default (2-4k) is far too small for graph extraction prompts."""
    keep_alive: str | int | None = None
    think: bool | str | None = None
    """Ollama ``think`` flag for reasoning models (``false`` disables thinking). Unset = model default."""
    options: dict[str, Any] = field(default_factory=dict)
    """Extra Ollama runtime options (``num_gpu``, ``repeat_penalty``, ...)."""

    # --- Hugging Face (in-process) only ----------------------------------
    device: str | None = None
    """``cuda``, ``cuda:1``, ``mps``, ``cpu`` or ``auto`` (default: spread over GPUs with accelerate)."""
    dtype: str = "auto"
    """``auto`` | ``bfloat16`` | ``float16`` | ``float32``."""
    quantization: str | None = None
    """``4bit`` or ``8bit`` (requires ``bitsandbytes``)."""
    trust_remote_code: bool = False
    batch_size: int = 1
    """>1 enables dynamic batching of concurrent requests into one ``generate`` call."""
    batch_wait_ms: int = 50
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs for ``AutoModelForCausalLM.from_pretrained``."""
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs for ``apply_chat_template`` (e.g. ``{"enable_thinking": false}`` for Qwen3)."""

    def __post_init__(self) -> None:
        self.provider = normalize_provider(self.provider)
        if not self.model:
            raise ConfigError("llm.model must be set")
        if self.structured_output not in ("auto", "json_schema", "json_object", "none"):
            raise ConfigError("llm.structured_output must be auto|json_schema|json_object|none")
        if self.concurrency < 1:
            raise ConfigError("llm.concurrency must be >= 1")
        if self.quantization not in (None, "4bit", "8bit"):
            raise ConfigError("llm.quantization must be 4bit, 8bit or null")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LLMConfig:
        return _from_dict(cls, data, "llm")


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""

    provider: str = "ollama"
    model: str = "bge-m3"
    base_url: str | None = None
    api_key: str | None = None

    dimensions: int | None = None
    """Embedding size. ``null`` = auto-detect by embedding a probe text."""
    send_dimensions: bool = False
    """Forward ``dimensions`` to the API (only for Matryoshka models that support it)."""
    max_tokens: int = 8192
    """Maximum input tokens the embedding model accepts."""
    batch_size: int = 32
    normalize: bool = True
    timeout: float = 600.0
    max_retries: int = 3
    concurrency: int = 4

    # asymmetric / instruction-tuned embedding models
    query_prefix: str = ""
    """Prepended to queries, e.g. ``"search_query: "`` (nomic) or ``"query: "`` (e5)."""
    document_prefix: str = ""
    """Prepended to documents, e.g. ``"search_document: "`` or ``"passage: "``."""
    query_instruction_template: str = "Instruct: {instruction}\nQuery: "
    """Used when a method passes a task instruction for queries (HippoRAG 2 does)."""

    # --- Ollama only ------------------------------------------------------
    num_ctx: int | None = None
    keep_alive: str | int | None = None
    truncate: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    # --- Hugging Face (in-process) only ----------------------------------
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    hf_backend: str = "auto"
    """``auto`` | ``sentence_transformers`` | ``transformers``."""
    pooling: str = "mean"
    """Pooling for the plain ``transformers`` backend: ``mean`` | ``cls`` | ``last``."""
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = normalize_provider(self.provider)
        if not self.model:
            raise ConfigError("embedding.model must be set")
        if self.hf_backend not in ("auto", "sentence_transformers", "transformers"):
            raise ConfigError("embedding.hf_backend must be auto|sentence_transformers|transformers")
        if self.pooling not in ("mean", "cls", "last"):
            raise ConfigError("embedding.pooling must be mean|cls|last")
        if self.batch_size < 1 or self.concurrency < 1:
            raise ConfigError("embedding.batch_size and embedding.concurrency must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EmbeddingConfig:
        return _from_dict(cls, data, "embedding")


@dataclass
class InputConfig:
    """Where the corpus comes from (see ``rag_common.documents``)."""

    path: str = "../data/sample"
    """A file or directory (relative paths are resolved against the config file)."""
    patterns: list[str] = field(
        default_factory=lambda: ["*.md", "*.markdown", "*.txt", "*.json", "*.jsonl", "*.csv"]
    )
    recursive: bool = True
    text_field: str = "text"
    title_field: str = "title"
    id_field: str = "id"
    encoding: str = "utf-8"
    limit: int | None = None
    """Only index the first N documents (handy for smoke tests)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InputConfig:
        return _from_dict(cls, data, "input")


# ---------------------------------------------------------------------------
# environment-variable expansion
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` inside strings."""
    if isinstance(value, str):

        def _sub(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name!r} is not set (referenced in config)")

        expanded = _ENV_PATTERN.sub(_sub, value)
        # an entirely-empty optional value means "unset"
        return None if expanded == "" and _ENV_PATTERN.fullmatch(value) else expanded
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def set_by_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    """``set_by_path(d, "llm.model", "x")`` -> ``d["llm"]["model"] = "x"``."""
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    """Resolve ``path`` relative to ``base_dir`` unless it is absolute (``~`` is expanded)."""
    p = Path(os.path.expanduser(str(path)))
    return p if p.is_absolute() else (base_dir / p).resolve()


def redact(data: Any) -> Any:
    """Return a copy of ``data`` with secrets masked (for printing configs)."""
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, str) and v and any(s in k.lower() for s in ("api_key", "token", "secret", "password")):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(data, list):
        return [redact(v) for v in data]
    return data
