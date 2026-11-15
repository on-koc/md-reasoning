# HippoRAG 2 indexing with local models

`index.py` builds a [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) index with an LLM
(for OpenIE: named-entity recognition + triple extraction) and an embedding model served by
**Hugging Face transformers**, an **OpenAI-compatible API** or **Ollama** (see the
[main README](../README.md) for the provider settings).

## Install

The `hipporag` wheel pins `torch==2.5.1`, `vllm==0.6.6` and `transformers==4.45.2` (CUDA-only,
Python <= 3.12). The adapter does not need vLLM, so install the runtime dependencies first and the
package itself without its pins:

```bash
python -m venv .venvs/hipporag
```

```bash
.venvs/hipporag/bin/pip install -r hipporag2/requirements.txt
```

```bash
.venvs/hipporag/bin/pip install --no-deps hipporag==2.0.0a4
```

`import hipporag` eagerly imports `vllm`, `gritlm`, `boto3` and `litellm`; any of them that is
not installed is replaced by a stub module at runtime (used code paths never touch them).
Install `sentence-transformers` as well for `provider: huggingface` embeddings.

## Run

```bash
.venvs/hipporag/bin/python hipporag2/index.py --check
```

```bash
.venvs/hipporag/bin/python hipporag2/index.py -i data/sample -o output/hipporag
```

HippoRAG indexes *passages* exactly as given, so `index.py` first splits documents into token
windows (`hipporag.chunk_token_size`, default 512 with 64 overlap, `title\npassage` format like
the HippoRAG benchmark corpora). A HippoRAG corpus file (`[{"title": ..., "text": ...}]`) can be
indexed as-is with `--set hipporag.chunk_token_size=0`.

Re-running is incremental: passages are keyed by content hash, OpenIE results are stored in
`openie_results_ner_<llm>.json` and every LLM response is cached in `llm_cache/`, so an
interrupted run resumes where it stopped. `--force` rebuilds everything.

Output: `output_dir/<llm>_<embedding>/` with `graph.pickle` (igraph), and
`{chunk,entity,fact}_embeddings/vdb_*.parquet`; `output_dir/openie_results_ner_<llm>.json`.

## Configuration

Besides the chunking keys, the `hipporag:` section accepts any field of HippoRAG's
`BaseConfig` (`hipporag/utils/config_utils.py`), e.g. `synonymy_edge_sim_threshold`,
`synonymy_edge_topk`, `passage_node_weight`, `linking_top_k`, `retrieval_top_k`, `damping`,
`qa_top_k`. `llm.max_tokens` maps to `max_new_tokens`; model names, URLs and `save_dir` come
from the `llm:` / `embedding:` / `output_dir` settings.

## How the models are plugged in (`hipporag_adapter.py`)

HippoRAG selects model classes by name (`_get_llm_class`, `_get_embedding_model_class`), which
only covers OpenAI models, vLLM, NV-Embed-v2, GritLM and Contriever. The adapter replaces both
factories with wrappers around `rag_common`:

* `LocalHippoLLM` implements HippoRAG's `BaseLLM.infer()` contract (`(text, metadata,
  cache_hit)`) and reuses HippoRAG's SQLite response cache. The OpenIE instance runs in JSON mode
  and only ever returns canonical JSON - HippoRAG parses OpenIE output with `eval()`, so raw model
  text is never handed to it.
* `LocalHippoEmbedding` implements `batch_encode(texts, instruction=..., norm=...)`; query-time
  task instructions are formatted with `embedding.query_instruction_template`.

## Retrieval / QA with the index

```python
import sys
from pathlib import Path
sys.path[:0] = ["hipporag2", "."]          # run from the repository root
from hipporag_adapter import build_hipporag
from rag_common import EmbeddingConfig, LLMConfig, create_embedder, create_llm

if __name__ == "__main__":                 # hipporag starts a multiprocessing manager on import
    llm = create_llm(LLMConfig(provider="ollama", model="qwen2.5:7b-instruct"))
    emb = create_embedder(EmbeddingConfig(provider="ollama", model="bge-m3"))
    hippo = build_hipporag(Path("output/hipporag"), llm, emb, {})
    print(hippo.retrieve(queries=["Who designed Skylark?"], num_to_retrieve=3))
    print(hippo.rag_qa(queries=["Who designed Skylark?"]))
```
