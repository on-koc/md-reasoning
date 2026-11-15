"""Run HippoRAG 2 with rag_common backends (Hugging Face / OpenAI-compatible / Ollama).

HippoRAG picks its LLM and embedding classes from the model *name*
(``_get_llm_class`` / ``_get_embedding_model_class``), which only covers OpenAI
models, vLLM, NV-Embed-v2, GritLM and Contriever. We replace those two factory
functions with ones that return thin wrappers around our backends, so every
provider and every model works, while the rest of HippoRAG (OpenIE, graph
construction, PPR retrieval, QA) stays untouched.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rag_common.embeddings import Embedder  # noqa: E402
from rag_common.llm import ChatLLM  # noqa: E402
from rag_common.parsing import extract_json  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# import shims
# ---------------------------------------------------------------------------


class _Unavailable:
    """Placeholder for optional heavy dependencies we never use."""

    _name = "dependency"

    def __init__(self, *args: Any, **kwargs: Any):
        raise ImportError(
            f"{self._name} is not installed. It is only needed for HippoRAG's built-in "
            f"offline/vLLM, GritLM or Bedrock code paths, which this adapter does not use."
        )


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def _stub(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__dict__["__hipporag_stub__"] = True
    sys.modules[name] = module
    return module


def install_import_shims() -> list[str]:
    """``import hipporag`` eagerly imports vllm, gritlm, boto3 and litellm even
    though our backends need none of them. Register stub modules for the ones
    that are not installed (vllm pins old torch/CUDA versions and is often
    impossible to install next to current libraries)."""
    stubbed: list[str] = []

    def unavailable(label: str) -> type:
        return type(f"Unavailable_{label}", (_Unavailable,), {"_name": label})

    if _missing("vllm"):
        _stub("vllm", LLM=unavailable("vllm"), SamplingParams=unavailable("vllm"))
        stubbed.append("vllm")
    if _missing("gritlm"):
        _stub("gritlm", GritLM=unavailable("gritlm"))
        stubbed.append("gritlm")
    if _missing("boto3"):
        _stub("boto3", client=unavailable("boto3"), Session=unavailable("boto3"))
        botocore = _stub("botocore")
        exceptions = _stub("botocore.exceptions", ClientError=type("ClientError", (Exception,), {}))
        botocore.exceptions = exceptions
        stubbed.append("boto3")
    if _missing("litellm"):
        _stub("litellm", completion=unavailable("litellm"))
        stubbed.append("litellm")
    if stubbed:
        logger.info("HippoRAG: using stub modules for unused optional deps: %s", ", ".join(stubbed))
    return stubbed


# ---------------------------------------------------------------------------
# LLM wrapper
# ---------------------------------------------------------------------------


def model_label(model: str) -> str:
    """Filesystem-safe label (HippoRAG uses the model name in directory/file names).

    Hub ids keep their org (``Qwen/Qwen2.5-7B`` -> ``Qwen_Qwen2.5-7B``); local
    checkpoint paths are reduced to their folder name.
    """
    path = Path(os.path.expanduser(model))
    if path.is_absolute() or model.startswith((".", "~")):
        model = path.name or model
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def _strict_json(text: str) -> str:
    """Canonical JSON or ``{}``.

    HippoRAG parses OpenIE output with ``eval()``; only ever handing it
    ``json.dumps`` output keeps model-generated text from being executed.
    """
    try:
        return json.dumps(extract_json(text), ensure_ascii=False)
    except ValueError:
        return "{}"


def make_llm_class():
    from hipporag.llm.base import BaseLLM
    from hipporag.llm.base import LLMConfig as HippoLLMConfig
    from hipporag.llm.openai_gpt import cache_response

    class LocalHippoLLM(BaseLLM):
        """HippoRAG ``BaseLLM`` backed by a rag_common :class:`ChatLLM`.

        ``infer`` keeps HippoRAG's contract (returns ``(text, metadata, cache_hit)``)
        and reuses HippoRAG's SQLite response cache, so interrupted runs resume.
        """

        def __init__(self, global_config: Any, backend: ChatLLM, *, json_output: bool = False):
            super().__init__(global_config)
            self.backend = backend
            self.json_output = json_output
            self.cache_dir = os.path.join(global_config.save_dir, "llm_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file_name = os.path.join(self.cache_dir, f"{model_label(self.llm_name)}_cache.sqlite")
            self._init_llm_config()

        def _init_llm_config(self) -> None:
            gc = self.global_config
            self.llm_config = HippoLLMConfig.from_dict(
                {
                    "llm_name": gc.llm_name,
                    "generate_params": {
                        "model": gc.llm_name,
                        "max_completion_tokens": gc.max_new_tokens,
                        "n": 1,
                        "seed": gc.seed,
                        "temperature": gc.temperature,
                    },
                }
            )

        @cache_response
        def infer(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, Any]]:
            params = {**self.llm_config.generate_params, **kwargs}
            result = self.backend.generate(
                messages,
                max_tokens=params.get("max_completion_tokens") or params.get("max_tokens"),
                temperature=params.get("temperature"),
                json_mode=self.json_output,
            )
            text = _strict_json(result.text) if self.json_output else result.text
            metadata = {
                "prompt_tokens": result.prompt_tokens or 0,
                "completion_tokens": result.completion_tokens or 0,
                "finish_reason": result.finish_reason or "stop",
            }
            return text, metadata

    return LocalHippoLLM


# ---------------------------------------------------------------------------
# embedding wrapper
# ---------------------------------------------------------------------------


def make_embedding_class():
    from hipporag.embedding_model.base import BaseEmbeddingModel
    from hipporag.embedding_model.base import EmbeddingConfig as HippoEmbeddingConfig

    class LocalHippoEmbedding(BaseEmbeddingModel):
        """HippoRAG ``BaseEmbeddingModel`` backed by a rag_common :class:`Embedder`."""

        def __init__(self, global_config: Any, backend: Embedder):
            super().__init__(global_config=global_config)
            self.backend = backend
            self.embedding_dim = backend.dim
            self.embedding_config = HippoEmbeddingConfig.from_dict(
                {
                    "embedding_model_name": self.embedding_model_name,
                    "norm": global_config.embedding_return_as_normalized,
                    "encode_params": {"batch_size": global_config.embedding_batch_size, "instruction": ""},
                }
            )

        def batch_encode(self, texts: list[str] | str, **kwargs: Any) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            instruction = kwargs.get("instruction") or None
            vectors = self.backend.embed(list(texts), is_query=bool(instruction), instruction=instruction)
            if kwargs.get("norm", self.embedding_config.norm):
                vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
            return vectors

    return LocalHippoEmbedding


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_hipporag(save_dir: Path, llm: ChatLLM, embedder: Embedder, options: dict[str, Any]):
    """Create a ``HippoRAG`` instance wired to our backends.

    ``options`` are HippoRAG ``BaseConfig`` fields (e.g. ``synonymy_edge_sim_threshold``).
    """
    install_import_shims()
    hippo_module = importlib.import_module("hipporag.HippoRAG")  # the module, not the class
    from hipporag.utils.config_utils import BaseConfig

    LocalHippoLLM = make_llm_class()
    LocalHippoEmbedding = make_embedding_class()

    valid = {f.name for f in dataclasses.fields(BaseConfig)}
    unknown = sorted(set(options) - valid)
    if unknown:
        raise ValueError(f"Unknown HippoRAG option(s) in 'hipporag:' section: {unknown}")
    blocked = sorted(set(options) & {"llm_name", "llm_base_url", "embedding_model_name", "embedding_base_url", "save_dir", "openie_mode"})
    if blocked:
        raise ValueError(f"'hipporag:' must not set {blocked}; use llm / embedding / output_dir instead")

    values: dict[str, Any] = {
        "save_dir": str(save_dir),
        "llm_name": model_label(llm.config.model),
        "embedding_model_name": model_label(embedder.config.model),
        "max_new_tokens": llm.config.max_tokens,
        "temperature": llm.config.temperature,
        "seed": llm.config.seed,
        "embedding_batch_size": embedder.config.batch_size,
        "embedding_return_as_normalized": embedder.config.normalize,
        "openie_mode": "online",
    }
    values.update(options)
    global_config = BaseConfig(**values)

    # swap HippoRAG's name-based factories for our wrappers
    hippo_module._get_llm_class = lambda config: LocalHippoLLM(config, llm)
    hippo_module._get_embedding_model_class = lambda embedding_model_name=None: (
        lambda global_config=None, embedding_model_name=None: LocalHippoEmbedding(global_config, embedder)
    )

    hippo = hippo_module.HippoRAG(global_config=global_config)
    # OpenIE (NER + triple extraction) answers in JSON: enforce JSON mode for it
    hippo.openie.llm_model = LocalHippoLLM(global_config, llm, json_output=True)
    return hippo
