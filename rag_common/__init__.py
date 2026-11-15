"""Shared, method-agnostic building blocks for the GraphRAG indexing scripts.

* :mod:`rag_common.config`     - ``llm`` / ``embedding`` / ``input`` config dataclasses
* :mod:`rag_common.llm`        - chat backends (huggingface, openai-compatible, ollama)
* :mod:`rag_common.embeddings` - embedding backends (same three providers)
* :mod:`rag_common.documents`  - corpus loading + chunking
* :mod:`rag_common.cli`        - CLI / YAML settings handling
"""

from .config import PROVIDERS, ConfigError, EmbeddingConfig, InputConfig, LLMConfig, normalize_provider
from .documents import Chunk, Document, chunk_documents, ensure_tiktoken_cache, load_documents
from .embeddings import Embedder, create_embedder
from .llm import ChatLLM, ChatResult, create_llm

__all__ = [
    "PROVIDERS",
    "ChatLLM",
    "ChatResult",
    "Chunk",
    "ConfigError",
    "Document",
    "Embedder",
    "EmbeddingConfig",
    "InputConfig",
    "LLMConfig",
    "chunk_documents",
    "create_embedder",
    "create_llm",
    "ensure_tiktoken_cache",
    "load_documents",
    "normalize_provider",
]
