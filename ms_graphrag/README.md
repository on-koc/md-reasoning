# MS GraphRAG indexing with local models

`index.py` builds a [Microsoft GraphRAG](https://github.com/microsoft/graphrag) (3.x) index
with an LLM and an embedding model served by **Hugging Face transformers**, an
**OpenAI-compatible API** or **Ollama** (see the [main README](../README.md) for the provider settings).

## Install

GraphRAG 3.x needs Python 3.11 - 3.13.

```bash
python -m venv .venvs/graphrag
```

```bash
.venvs/graphrag/bin/pip install -r ms_graphrag/requirements.txt
```

Add `-r requirements-hf.txt` for `provider: huggingface`.

## Run

```bash
.venvs/graphrag/bin/python ms_graphrag/index.py --check
```

```bash
.venvs/graphrag/bin/python ms_graphrag/index.py -i data/sample -o output/graphrag
```

| Option | Effect |
|---|---|
| `--method standard\|fast` | `standard`: the LLM extracts the graph. `fast`: NLP noun-phrase extraction, LLM only for community reports |
| `--update` | incremental update: only documents not yet in the index are added (GraphRAG's update mode cannot modify or delete documents - re-run without `--update` for that) |
| `--init-only` | create the project (`settings.yaml`, `prompts/`) and stop, e.g. to edit prompts first |
| `--no-cache` | ignore GraphRAG's LLM cache (by default re-runs reuse cached responses) |
| `--skip-validation` | skip the connectivity test before indexing |
| `-m configs/vllm.yaml`, `--set ...`, `--limit N`, `--print-config` | see the main README |

## What the script does

1. Creates a GraphRAG project in `output_dir` on the first run (same as `graphrag init`:
   `settings.yaml`, `.env`, `prompts/`). Edit those files freely afterwards.
2. Registers a custom GraphRAG model type **`local`** (`graphrag_adapter.py`) and fills the
   `completion_models` / `embedding_models` entries from the `llm:` / `embedding:` sections of
   `config.yaml` on every run.
3. Probes the embedding model to set `vector_store.vector_size` (GraphRAG defaults to 3072 and
   LanceDB rejects vectors of any other size).
4. Merges `graphrag.settings` from `config.yaml` over `settings.yaml` (chunk size, entity types,
   gleanings, report lengths, ...; any [GraphRAG setting](https://microsoft.github.io/graphrag/config/yaml/)).
5. Loads the corpus with `rag_common` (`.md/.txt/.json/.jsonl/.csv`) and runs
   `graphrag.api.build_index(...)`, then prints the row counts of the output tables.

Output: `output_dir/output/*.parquet` (documents, text_units, entities, relationships,
communities, community_reports), `output_dir/output/lancedb/` (vectors), `cache/`, `logs/`.

## How the models are plugged in

GraphRAG 3 routes every model call through `graphrag-llm`. `graphrag_adapter.py` implements its
`LLMCompletion` and `LLMEmbedding` interfaces on top of `rag_common`, so GraphRAG's own
middleware (cache, retries, rate limiting, metrics) still applies. Details:

* Community reports request a pydantic `response_format`. It is translated to JSON-schema
  constrained decoding (Ollama `format`, OpenAI `response_format`, or a schema hint for
  Hugging Face models); the output is repaired/validated inside the retry loop, so a malformed
  answer is retried (once with sampling) instead of being cached.
* The cache key includes the backend and model name, so switching models never reuses another
  model's answers.
* Token budgeting uses tiktoken (`graphrag.encoding_model`, default `o200k_base`).
* `concurrent_requests` follows `llm.concurrency`.

Any `settings.yaml` model entry can use the `local` type directly, with every `rag_common`
option as an extra key:

```yaml
completion_models:
  default_completion_model:
    type: local
    model_provider: openai          # huggingface | openai | ollama
    model: Qwen/Qwen2.5-14B-Instruct
    api_base: http://localhost:8000/v1
    concurrency: 16
```

## Output format vs. GraphRAG 2.x

The tables are exactly what GraphRAG 3.3's own `graphrag index` writes (same columns, ids and
values for the same input). Compared with GraphRAG **2.4.0** (verified by indexing the same corpus
with both versions):

| Output | Change in 3.x |
|---|---|
| `documents.parquet` | `metadata` column renamed to `raw_data` |
| `text_units.parquet` | `document_ids` (list) replaced by `document_id` (string); ids and `n_tokens` differ (chunking now uses `o200k_base` instead of `cl100k_base`) |
| `entities.parquet` | `x`, `y` (UMAP layout, `0.0`/NaN unless UMAP was enabled) removed |
| `relationships.parquet` | same columns; `source`/`target` may be swapped for some edges (3.x keeps the LLM's order) |
| `communities.parquet`, `community_reports.parquet` | same columns; community membership can differ (new Leiden implementation) |
| `output/lancedb/` | tables renamed (`default-entity-description` -> `entity_description`, ...); columns `id, text, vector, attributes` -> `id, vector` + date fields |
| all parquet files | strings stored as Arrow `large_string` instead of `string` (transparent for pandas / polars / DuckDB) |

Entity / relationship / community ids are random UUIDs in both versions, so they never match
across runs. Code written for 2.x outputs needs the column renames above; query code that reads
the LanceDB tables directly needs adapting.

## Notes

* GraphRAG's CLI (`graphrag index/query/prompt-tune`) does not know the `local` type. Use these
  scripts, or call `graphrag_adapter.register()` before using the GraphRAG Python API in your own code.
* Small local models often produce fewer, noisier entities. Prefer 14B+ instruct models, keep
  `max_gleanings` at 1 or more, and consider tuning `prompts/extract_graph.txt` to your domain.
