#!/usr/bin/env python
"""Build a HippoRAG 2 index with local models.

The LLM (used for OpenIE: NER + triple extraction) and the embedding model can be served by
  * huggingface - transformers / sentence-transformers inside this process
  * openai      - any OpenAI-compatible server (vLLM, SGLang, llama.cpp, LM Studio, TGI, ...)
  * ollama      - Ollama's native API

HippoRAG indexes *passages*; documents are split into token windows first
(``hipporag.chunk_token_size``). Re-running is incremental: known passages are
skipped and cached OpenIE results / LLM responses are reused.

Examples
  python hipporag2/index.py                                  # hipporag2/config.yaml
  python hipporag2/index.py -i ./corpus.json -o ./index/hipporag -m configs/vllm.yaml
  python hipporag2/index.py --limit 20 --set hipporag.chunk_token_size=256
  python hipporag2/index.py --check                          # test model connectivity
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rag_common.cli import build_parser, handle_common_actions, load_settings, log_usage, run, setup_logging  # noqa: E402
from rag_common.documents import chunk_documents, load_documents  # noqa: E402

logger = logging.getLogger("hipporag2.index")


def parse_args():
    parser = build_parser(__doc__, HERE / "config.yaml")
    g = parser.add_argument_group("HippoRAG 2")
    g.add_argument("--force", action="store_true", help="rebuild the index and redo OpenIE from scratch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    settings = load_settings(args, "hipporag")
    handle_common_actions(args, settings)

    options = dict(settings.method)
    chunk_size = options.pop("chunk_token_size", 512)
    overlap = options.pop("chunk_overlap_token_size", 64)
    chunk_by = options.pop("chunk_by", "token")
    chunk_encoding = options.pop("chunk_encoding", "cl100k_base")
    prepend_title = options.pop("prepend_title", True)
    if args.force:
        options.update(force_index_from_scratch=True, force_openie_from_scratch=True)

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
    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        overlap=overlap,
        by=chunk_by,
        encoding_name=chunk_encoding,
        prepend_title=prepend_title,
    )
    passages = list(dict.fromkeys(c.text for c in chunks))  # HippoRAG keys passages by content hash
    logger.info("%d documents -> %d passages (chunk_size=%s, overlap=%s)", len(docs), len(passages), chunk_size, overlap)

    from hipporag_adapter import build_hipporag  # imports hipporag lazily (it starts a multiprocessing manager)
    from rag_common.embeddings import create_embedder
    from rag_common.llm import create_llm

    llm = create_llm(settings.llm)
    embedder = create_embedder(settings.embedding)
    logger.info("LLM: %s | embeddings: %s (%d-d)", llm.name, embedder.name, embedder.dim)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    hippo = build_hipporag(settings.output_dir, llm, embedder, options)

    t0 = time.perf_counter()
    hippo.index(docs=passages)
    logger.info("HippoRAG indexing finished in %.1fs -> %s", time.perf_counter() - t0, hippo.working_dir)
    logger.info(
        "passages: %d | entities: %d | facts: %d | graph: %d nodes, %d edges",
        len(hippo.chunk_embedding_store.get_all_ids()),
        len(hippo.entity_embedding_store.get_all_ids()),
        len(hippo.fact_embedding_store.get_all_ids()),
        hippo.graph.vcount(),
        hippo.graph.ecount(),
    )
    log_usage(llm, embedder)
    return 0


if __name__ == "__main__":  # required: hipporag spawns a multiprocessing manager on import
    run(main)
