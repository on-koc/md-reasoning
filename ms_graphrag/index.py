#!/usr/bin/env python
"""Build a Microsoft GraphRAG (3.x) index with local models.

The LLM and the embedding model can be served by
  * huggingface - transformers / sentence-transformers inside this process
  * openai      - any OpenAI-compatible server (vLLM, SGLang, llama.cpp, LM Studio, TGI, ...)
  * ollama      - Ollama's native API

The output directory is a regular GraphRAG project (settings.yaml, prompts/,
output/*.parquet, output/lancedb, cache/, logs/), created on the first run.

Examples
  python ms_graphrag/index.py                                  # ms_graphrag/config.yaml
  python ms_graphrag/index.py -i ./docs -o ./index/graphrag -m configs/vllm.yaml
  python ms_graphrag/index.py --method fast --limit 10         # quick NLP-based run
  python ms_graphrag/index.py --update                         # incremental update
  python ms_graphrag/index.py --check                          # test model connectivity
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rag_common.cli import (  # noqa: E402
    run,
    Settings,
    build_parser,
    handle_common_actions,
    load_settings,
    log_usage,
    setup_logging,
)
from rag_common.config import ConfigError  # noqa: E402
from rag_common.documents import Document, load_documents  # noqa: E402

logger = logging.getLogger("ms_graphrag.index")

COMPLETION_ID = "default_completion_model"
EMBEDDING_ID = "default_embedding_model"

MODELS_BLOCK = """\
### LLM settings ###
## Managed by ms_graphrag/index.py: these entries are completed at runtime from the
## `llm:` / `embedding:` sections of the index config (type `local` = rag_common
## backends: huggingface | openai | ollama). Extra GraphRAG options added here
## (e.g. rate_limit) are merged with the runtime values.

completion_models:
  {completion_id}:
    type: local
    model_provider: {llm_provider}
    model: {llm_model}

embedding_models:
  {embedding_id}:
    type: local
    model_provider: {emb_provider}
    model: {emb_model}

"""


def _deep_merge(dest: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _deep_merge(dest[key], value)
        else:
            dest[key] = copy.deepcopy(value)
    return dest


def parse_args():
    parser = build_parser(__doc__, HERE / "config.yaml")
    g = parser.add_argument_group("GraphRAG")
    g.add_argument("--method", choices=["standard", "fast"], help="indexing method (overrides graphrag.method)")
    g.add_argument("--update", action="store_true", help="incrementally update an existing index")
    g.add_argument("--no-cache", action="store_true", help="disable GraphRAG's LLM response cache")
    g.add_argument("--skip-validation", action="store_true", help="skip the model connectivity test before indexing")
    g.add_argument("--init-only", action="store_true", help="only create the project (settings.yaml + prompts) and exit")
    g.add_argument("--verbose", action="store_true", help="verbose GraphRAG progress output")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# project + configuration
# ---------------------------------------------------------------------------


def ensure_project(root: Path, settings: Settings) -> None:
    """Create a GraphRAG project (settings.yaml, prompts/) in ``root`` if missing."""
    settings_file = root / "settings.yaml"
    if settings_file.exists():
        return
    from graphrag.cli.initialize import initialize_project_at

    logger.info("Initializing GraphRAG project in %s", root)
    initialize_project_at(root, force=False, model=settings.llm.model, embedding_model=settings.embedding.model)
    text = settings_file.read_text(encoding="utf-8")
    block = MODELS_BLOCK.format(
        completion_id=COMPLETION_ID,
        embedding_id=EMBEDDING_ID,
        llm_provider=settings.llm.provider,
        llm_model=_yaml_scalar(settings.llm.model),
        emb_provider=settings.embedding.provider,
        emb_model=_yaml_scalar(settings.embedding.model),
    )
    start = text.find("### LLM settings ###")
    end = text.find("### Document processing settings ###")
    if start != -1 and end > start:
        text = text[:start] + block + text[end:]
    else:  # template changed upstream: fall back to a structural rewrite
        import yaml

        data = yaml.safe_load(text)
        data.update(yaml.safe_load(block))
        text = yaml.safe_dump(data, sort_keys=False)
    settings_file.write_text(text, encoding="utf-8")
    (root / ".env").write_text("# Environment variables referenced from settings.yaml (optional)\n", encoding="utf-8")


def _yaml_scalar(value: str) -> str:
    import json

    # JSON strings are valid YAML; "$" must be doubled for GraphRAG's env-var templating
    return json.dumps(value).replace("$", "$$")


def model_entries(settings: Settings) -> dict[str, Any]:
    """completion_models / embedding_models entries for the ``local`` model type.

    Every rag_common config field is forwarded unchanged, so the adapter
    rebuilds exactly the same config (and reuses the same model instance).
    """
    from graphrag_adapter import LOCAL_TYPE

    retry = {"type": "exponential_backoff", "max_retries": int(settings.method.get("max_retries", 5))}
    encoding = settings.method.get("encoding_model", "o200k_base")

    def entry(cfg: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(cfg)
        out: dict[str, Any] = {
            "type": LOCAL_TYPE,
            "model_provider": cfg.pop("provider"),
            "model": cfg.pop("model"),
        }
        base_url, api_key = cfg.pop("base_url"), cfg.pop("api_key")
        if base_url:
            out["api_base"] = base_url
        if api_key:
            out["api_key"] = api_key
        out.update(cfg)
        out["encoding_model"] = encoding
        out["retry"] = dict(retry)
        return out

    return {
        "completion_models": {COMPLETION_ID: entry(dataclasses.asdict(settings.llm))},
        "embedding_models": {EMBEDDING_ID: entry(dataclasses.asdict(settings.embedding))},
    }


def build_overrides(settings: Settings, vector_size: int) -> dict[str, Any]:
    overrides: dict[str, Any] = model_entries(settings)
    overrides["concurrent_requests"] = settings.llm.concurrency
    for section in ("embed_text",):
        overrides.setdefault(section, {})["embedding_model_id"] = EMBEDDING_ID
    for section in ("extract_graph", "summarize_descriptions", "extract_claims", "community_reports"):
        overrides.setdefault(section, {})["completion_model_id"] = COMPLETION_ID
    user = settings.method.get("settings") or {}
    if not isinstance(user, dict):
        raise ValueError("graphrag.settings must be a mapping of GraphRAG settings")
    _deep_merge(overrides, user)
    configured = (overrides.get("vector_store") or {}).get("vector_size")
    if configured and configured != vector_size:
        logger.warning("Overriding vector_store.vector_size=%s with detected embedding size %s", configured, vector_size)
    overrides.setdefault("vector_store", {})["vector_size"] = vector_size
    return overrides


def documents_frame(docs: list[Document]):
    """The ``documents`` table in the exact shape GraphRAG's own input reader produces."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "id": graphrag_doc_id(d),
                "human_readable_id": i,
                "title": doc_title(d),
                "text": d.text,
                "creation_date": creation_date(d),
                "raw_data": None,
            }
            for i, d in enumerate(docs)
        ]
    )


def graphrag_doc_id(doc: Document) -> str:
    """GraphRAG's document id: SHA-512 of the text (as in GraphRAG 2.x and 3.x).
    Explicit ids from JSON/CSV records (``input.id_field``) are kept."""
    if doc.id.startswith("doc-"):  # content-derived id from rag_common
        from graphrag_input.hashing import gen_sha512_hash

        return gen_sha512_hash({"text": doc.text}, ["text"])
    return doc.id


def creation_date(doc: Document) -> str:
    """Source-file ctime in GraphRAG's format, e.g. ``2025-07-15 09:30:00 +0000``."""
    ts = datetime.fromtimestamp(doc.created, tz=timezone.utc) if doc.created else datetime.now(timezone.utc)
    return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def doc_title(doc: Document) -> str:
    """GraphRAG document title: the relative file path for files (unique, like
    GraphRAG's own file reader), the record title for JSON/CSV records."""
    if doc.source and "#" not in doc.source:
        return doc.source
    return doc.title or doc.source or doc.id


def new_documents_for_update(docs: list[Document], output_dir: Path) -> list[Document]:
    """GraphRAG's incremental update only *adds* documents it has not seen
    (matched by title). Its own filter is skipped when documents are passed in
    directly, so we apply the same rule here (plus an id check)."""
    import pandas as pd

    previous_file = output_dir / "documents.parquet"
    if not previous_file.exists():
        raise ConfigError(f"--update needs an existing index in {output_dir}; run once without --update first")
    previous = pd.read_parquet(previous_file, columns=["id", "title"])
    known_ids, known_titles = set(previous["id"]), set(previous["title"])
    new, changed = [], []
    for doc in docs:
        if graphrag_doc_id(doc) in known_ids:
            continue
        if doc_title(doc) in known_titles:
            changed.append(doc_title(doc))
            continue
        new.append(doc)
    if changed:
        logger.warning(
            "%d document(s) changed since they were indexed and are NOT updated (GraphRAG's incremental "
            "mode only adds documents; re-run without --update to rebuild): %s",
            len(changed), ", ".join(changed[:10]) + (" ..." if len(changed) > 10 else ""),
        )
    return new


def detect_vector_size(settings: Settings) -> int:
    """Embedding size for the LanceDB schema (GraphRAG defaults to 3072)."""
    from rag_common.embeddings import create_embedder

    embedder = create_embedder(settings.embedding)
    dim = embedder.dim
    logger.info("Embedding model %s -> %d dimensions", embedder.name, dim)
    return dim


def summarize_outputs(output_dir: Path) -> None:
    import pandas as pd

    for name in ("documents", "text_units", "entities", "relationships", "communities", "community_reports", "covariates"):
        path = output_dir / f"{name}.parquet"
        if path.exists():
            logger.info("  %-18s %6d rows  (%s)", name, len(pd.read_parquet(path)), path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    settings = load_settings(args, "graphrag")
    if args.method:
        settings.method["method"] = args.method
    handle_common_actions(args, settings)

    import graphrag_adapter  # lives next to this script

    graphrag_adapter.register()

    root = settings.output_dir
    root.mkdir(parents=True, exist_ok=True)
    ensure_project(root, settings)
    if args.init_only:
        logger.info("Project ready: %s (edit settings.yaml / prompts/ and run again)", root)
        return 0

    from rag_common.embeddings import create_embedder
    from rag_common.llm import create_llm

    if not args.skip_validation:
        llm = create_llm(settings.llm)
        t0 = time.perf_counter()
        reply = llm.generate("Reply with the single word OK.", max_tokens=8)
        logger.info("LLM %s reachable (%.1fs): %r", llm.name, time.perf_counter() - t0, reply.text[:40])
    vector_size = detect_vector_size(settings)

    docs = load_documents(
        settings.input_path,
        patterns=settings.input.patterns,
        recursive=settings.input.recursive,
        text_field=settings.input.text_field,
        title_field=settings.input.title_field,
        id_field=settings.input.id_field,
        encoding=settings.input.encoding,
        limit=settings.input.limit,
    )

    from graphrag import api
    from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
    from graphrag.config.load_config import load_config
    from graphrag_cache.cache_type import CacheType

    overrides = build_overrides(settings, vector_size)
    config = load_config(root_dir=root, cli_overrides=overrides)  # note: chdir()s into the project root
    if args.no_cache:
        config.cache.type = CacheType.Noop
    if args.update:
        docs = new_documents_for_update(docs, Path(config.output_storage.base_dir))
        if not docs:
            logger.info("No new documents - the index is up to date.")
            return 0

    method = settings.method.get("method", "standard")
    logger.info(
        "Indexing %d documents with GraphRAG (%s%s) into %s", len(docs), method, ", update" if args.update else "", root
    )
    t0 = time.perf_counter()
    outputs = asyncio.run(
        api.build_index(
            config=config,
            method=method,
            is_update_run=args.update,
            callbacks=[ConsoleWorkflowCallbacks(verbose=args.verbose)],
            verbose=args.verbose,
            input_documents=documents_frame(docs),
        )
    )
    failed = [o for o in outputs if o.error is not None]
    for o in failed:
        logger.error("workflow %s failed: %s", o.workflow, o.error)
    logger.info("GraphRAG indexing finished in %.1fs (%d workflows, %d failed)", time.perf_counter() - t0, len(outputs), len(failed))
    summarize_outputs(Path(config.output_storage.base_dir))
    log_usage(create_llm(settings.llm), create_embedder(settings.embedding))
    return 1 if failed else 0


if __name__ == "__main__":
    run(main)
