# LightRAG indexing with local models

`index.py` builds a [LightRAG](https://github.com/HKUDS/LightRAG) index with an LLM and an
embedding model served by **Hugging Face transformers**, an **OpenAI-compatible API** or
**Ollama** (see the [main README](../README.md) for the provider settings).

## Install

```bash
python -m venv .venvs/lightrag
```

```bash
.venvs/lightrag/bin/pip install -r lightrag/requirements.txt
```

Add `-r requirements-hf.txt` for `provider: huggingface`.

## Run

```bash
.venvs/lightrag/bin/python lightrag/index.py --check
```

```bash
.venvs/lightrag/bin/python lightrag/index.py -i data/sample -o output/lightrag
```

Re-running the same command is incremental:

* documents already in the index (same content hash) are skipped,
* failed or interrupted documents are retried (LightRAG's pipeline resume),
* a file whose content changed is deleted from the index and inserted again.

`--workspace NAME` keeps several independent indexes in one `output_dir`.

Output (`output_dir`): `graph_chunk_entity_relation.graphml` (knowledge graph),
`vdb_{chunks,entities,relationships}.json` (vectors), `kv_store_*.json` (documents, chunks,
LLM cache, document status).

## Configuration

The `lightrag:` section of `config.yaml` is passed straight to `LightRAG(...)`, so every
constructor argument can be set there (chunk sizes, `entity_extract_max_gleaning`,
`entity_extraction_use_json`, `max_parallel_insert`, `addon_params`, storage backends such as
`graph_storage: Neo4JStorage`, ...). The script fills in `working_dir`, `llm_model_func`,
`llm_model_name`, `embedding_func`, and derives `llm_model_max_async`,
`embedding_func_max_async`, `embedding_batch_num` and the LLM / embedding timeouts from the
`llm:` / `embedding:` sections unless they are set explicitly.

`entity_extraction_use_json: true` switches extraction to JSON output, which the backends
enforce natively (Ollama `format`, OpenAI `response_format`); this often helps smaller models.

## How the models are plugged in (`lightrag_adapter.py`)

* `make_llm_model_func(llm)` returns the async `llm_model_func(prompt, system_prompt,
  history_messages, **kwargs)` LightRAG expects. `response_format` / `keyword_extraction` /
  `entity_extraction` are mapped to JSON(-schema) mode; answers cut off by `max_tokens` are
  returned as `TruncatedResponse` so LightRAG does not cache them.
* `make_embedding_func(embedder)` returns an `EmbeddingFunc` with the auto-detected dimension.
  It is asymmetric-aware: LightRAG passes `context="query"` for searches, which applies
  `embedding.query_prefix` instead of `embedding.document_prefix`.

## Querying the index

Use the same adapters so the query side talks to the same models:

```python
import asyncio, sys
sys.path[:0] = ["lightrag", "."]          # run from the repository root
from lightrag import LightRAG, QueryParam
from lightrag_adapter import make_embedding_func, make_llm_model_func
from rag_common import EmbeddingConfig, LLMConfig, create_embedder, create_llm

async def main():
    llm = create_llm(LLMConfig(provider="ollama", model="qwen2.5:7b-instruct"))
    emb = create_embedder(EmbeddingConfig(provider="ollama", model="bge-m3"))
    rag = LightRAG(working_dir="output/lightrag", llm_model_func=make_llm_model_func(llm),
                   llm_model_name=llm.config.model, embedding_func=make_embedding_func(emb))
    await rag.initialize_storages()
    print(await rag.aquery("Who designed Skylark?", param=QueryParam(mode="mix", enable_rerank=False)))
    await rag.finalize_storages()

asyncio.run(main())
```

## Notes

* LightRAG stores documents under their file *basename*; nested files are registered as
  `sub__dir__file.md` so equal names in different folders do not collide.
* LightRAG chunks with tiktoken. On offline machines set `TIKTOKEN_CACHE_DIR` (or install
  `litellm`, whose bundled encodings are then used automatically).
