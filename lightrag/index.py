#!/usr/bin/env python
"""Build a LightRAG index with local models.

The LLM and the embedding model can be served by
  * huggingface - transformers / sentence-transformers inside this process
  * openai      - any OpenAI-compatible server (vLLM, SGLang, llama.cpp, LM Studio, TGI, ...)
  * ollama      - Ollama's native API

Re-running is incremental: documents that are already processed are skipped
and failed / interrupted ones are retried.

Examples
  python lightrag/index.py                                   # lightrag/config.yaml
  python lightrag/index.py -i ./docs -o ./index/lightrag -m configs/huggingface.yaml
  python lightrag/index.py --limit 5 --log-level DEBUG
  python lightrag/index.py --check                           # test model connectivity
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import sys
import time
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
from rag_common.documents import ensure_tiktoken_cache, load_documents  # noqa: E402

logger = logging.getLogger("lightrag_index")

# keys of the `lightrag:` config section that are handled by this script
# (everything else is passed straight to the LightRAG constructor)
_MANAGED = {"working_dir", "llm_model_func", "llm_model_name", "embedding_func"}


def parse_args():
    parser = build_parser(__doc__, HERE / "config.yaml")
    g = parser.add_argument_group("LightRAG")
    g.add_argument("--workspace", help="LightRAG workspace (isolates several indexes in one output_dir)")
    return parser.parse_args()


def lightrag_kwargs(settings: Settings, llm_func, embedding_func) -> dict[str, Any]:
    from lightrag import LightRAG

    options = dict(settings.method)
    valid = {f.name for f in dataclasses.fields(LightRAG) if f.init} | {"addon_params"}
    unknown = sorted(set(options) - valid)
    if unknown:
        raise ValueError(f"Unknown LightRAG option(s) in 'lightrag:' section: {unknown}")
    managed = sorted(set(options) & _MANAGED)
    if managed:
        raise ValueError(f"'lightrag:' must not set {managed}; use output_dir / llm / embedding instead")

    kwargs: dict[str, Any] = {
        "working_dir": str(settings.output_dir),
        "llm_model_func": llm_func,
        "llm_model_name": settings.llm.model,
        "llm_model_max_async": settings.llm.concurrency,
        "embedding_func": embedding_func,
        "embedding_func_max_async": settings.embedding.concurrency,
        "embedding_batch_num": settings.embedding.batch_size,
        # LightRAG's defaults (240s LLM / 30s embedding) are too tight for local models
        "default_llm_timeout": int(settings.llm.timeout),
        "default_embedding_timeout": int(settings.embedding.timeout),
    }
    kwargs.update(options)  # explicit LightRAG options win
    return kwargs


def source_key(doc) -> str:
    """LightRAG identifies documents by file *basename*; flatten nested paths
    (``a/README.md`` -> ``a__README.md``) so equal names in different folders don't collide."""
    return (doc.source or doc.title or doc.id).replace("\\", "/").strip("/").replace("/", "__")


def _field(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, dict) else getattr(record, name, None)


async def plan_changes(rag, docs) -> tuple[list, list[str]]:
    """Split ``docs`` into new documents and ids of outdated versions to delete.

    Documents whose id (content hash) is already known are skipped - LightRAG
    resumes pending/failed ones by itself. A known file name with new content
    means the file changed: the old version is deleted and the new one inserted.
    """
    from lightrag.base import DocStatus

    existing = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
    by_key = {
        _field(rec, "file_path"): doc_id
        for doc_id, rec in existing.items()
        if not str(doc_id).startswith("dup-") and _field(rec, "file_path")
    }
    new, outdated = [], []
    for doc in docs:
        if doc.id in existing:
            continue
        old = by_key.get(source_key(doc))
        if old and old != doc.id:
            outdated.append(old)
        new.append(doc)
    return new, outdated


async def index_documents(settings: Settings, docs) -> int:
    from collections import Counter

    from lightrag import LightRAG

    from lightrag_adapter import make_embedding_func, make_llm_model_func
    from rag_common.embeddings import create_embedder
    from rag_common.llm import create_llm

    llm = create_llm(settings.llm)
    embedder = create_embedder(settings.embedding)
    logger.info("LLM: %s | embeddings: %s (%d-d)", llm.name, embedder.name, embedder.dim)

    rag = LightRAG(**lightrag_kwargs(settings, make_llm_model_func(llm), make_embedding_func(embedder)))
    await rag.initialize_storages()
    try:
        t0 = time.perf_counter()
        new, outdated = await plan_changes(rag, docs)
        logger.info(
            "%d documents: %d new, %d already indexed, %d changed",
            len(docs), len(new), len(docs) - len(new), len(outdated),
        )
        for old_id in outdated:
            result = await rag.adelete_by_doc_id(old_id)
            logger.info("Removed outdated version %s: %s", old_id, getattr(result, "status", result))
        if new:
            await rag.ainsert(
                [d.text for d in new],
                ids=[d.id for d in new],
                file_paths=[source_key(d) for d in new],
            )
        else:
            await rag.apipeline_process_enqueue_documents()  # resume pending / failed documents
        elapsed = time.perf_counter() - t0

        records = await rag.doc_status.get_by_ids([d.id for d in docs])
        statuses: Counter = Counter()
        failed = 0
        for doc, rec in zip(docs, records):
            status = str(_field(rec, "status") or "missing") if rec else "missing"
            status = status.split(".")[-1].lower()
            statuses[status] += 1
            if status in ("failed", "missing"):
                failed += 1
                logger.error("FAILED %s (%s): %s", doc.id, doc.source, _field(rec, "error_msg") if rec else "not in doc status")
        logger.info("Finished in %.1fs. Document status: %s", elapsed, dict(statuses))
        await _log_graph_stats(rag)
    finally:
        await rag.finalize_storages()
    log_usage(llm, embedder)
    return 1 if failed else 0


async def _log_graph_stats(rag) -> None:
    try:
        graph = rag.chunk_entity_relation_graph
        labels = await graph.get_all_labels()
        edges = await graph.get_all_edges() if hasattr(graph, "get_all_edges") else None
        logger.info("Knowledge graph: %d entities%s", len(labels), f", {len(edges)} relations" if edges is not None else "")
    except Exception as exc:  # noqa: BLE001 - statistics are best effort
        logger.debug("could not compute graph statistics: %s", exc)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    settings = load_settings(args, "lightrag")
    if args.workspace is not None:
        settings.method["workspace"] = args.workspace
    handle_common_actions(args, settings)
    ensure_tiktoken_cache()  # LightRAG chunks with tiktoken; works offline if litellm is installed

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
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(index_documents(settings, docs))


if __name__ == "__main__":
    run(main)
