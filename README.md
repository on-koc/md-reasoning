# md-reasoning

Indexing scripts for three graph-based RAG methods, all running on **local models**:

| Folder | Method | Upstream |
|---|---|---|
| [`ms_graphrag/`](ms_graphrag/) | Microsoft GraphRAG 3.x | <https://github.com/microsoft/graphrag> |
| [`lightrag/`](lightrag/) | LightRAG | <https://github.com/HKUDS/LightRAG> |
| [`hipporag2/`](hipporag2/) | HippoRAG 2 | <https://github.com/OSU-NLP-Group/HippoRAG> |

Each method can use any of three model providers, for the LLM and the embedding model independently:

| `provider` | How the model is called | Typical servers / models |
|---|---|---|
| `huggingface` | In-process with `transformers` (chat) and `sentence-transformers` / `transformers` (embeddings) | Any Hub id or local checkpoint, e.g. `Qwen/Qwen2.5-7B-Instruct`, `BAAI/bge-m3` |
| `openai` | `openai` Python package against an OpenAI-compatible API | vLLM, SGLang, llama.cpp `llama-server`, LM Studio, TGI/TEI, LocalAI, Ollama's `/v1` |
| `ollama` | `ollama` Python package against Ollama's native API | `qwen2.5:7b-instruct`, `llama3.1:8b`, `bge-m3`, `nomic-embed-text`, ... |

## Layout

```
rag_common/            shared code used by all three methods
  config.py            llm / embedding / input settings (YAML + env vars + CLI)
  llm.py               chat backends: huggingface | openai | ollama
  embeddings.py        embedding backends: huggingface | openai | ollama
  documents.py         corpus loading (.md .txt .json .jsonl .csv) + chunking
  cli.py               shared command-line options
configs/               ready-made model profiles (ollama, vllm, huggingface, ...)
ms_graphrag/           index.py, graphrag_adapter.py, config.yaml, requirements.txt, README.md
lightrag/              index.py, lightrag_adapter.py, config.yaml, requirements.txt, README.md
hipporag2/             index.py, hipporag_adapter.py, config.yaml, requirements.txt, README.md
data/sample/           three small Markdown documents for a first run
tests/                 mock OpenAI/Ollama server, smoke test, tiny-HF-model generator
```

Each `<method>/<method>_adapter.py` plugs the shared backends into the framework through
its own extension point (GraphRAG: a custom `graphrag-llm` model type; LightRAG:
`llm_model_func` + `EmbeddingFunc`; HippoRAG: replacement `BaseLLM` /
`BaseEmbeddingModel` classes), so the frameworks run unmodified.

## Quick start

The three frameworks pin conflicting dependencies, so use one virtualenv per method:

```bash
python -m venv .venvs/lightrag
```

```bash
.venvs/lightrag/bin/pip install -r lightrag/requirements.txt
```

Start the models (here with Ollama) and run the indexer:

```bash
ollama pull qwen2.5:7b-instruct && ollama pull bge-m3
```

```bash
.venvs/lightrag/bin/python lightrag/index.py --check
```

```bash
.venvs/lightrag/bin/python lightrag/index.py -i data/sample -o output/lightrag
```

The other methods work the same way (`ms_graphrag/index.py`, `hipporag2/index.py`);
see each folder's README for its install notes (HippoRAG 2 needs a `--no-deps` install).
For `provider: huggingface` also install [`requirements-hf.txt`](requirements-hf.txt).

## Choosing models and providers

Every script reads `<method>/config.yaml`. Three ways to change the models:

```bash
# 1) a profile from configs/ replaces the llm: and embedding: sections
python ms_graphrag/index.py -m configs/vllm.yaml

# 2) dedicated flags
python hipporag2/index.py --llm-provider openai --llm-model Qwen/Qwen2.5-7B-Instruct \
    --llm-base-url http://localhost:8000/v1 --embedding-provider huggingface --embedding-model BAAI/bge-m3

# 3) any key with --set (dotted path, YAML value)
python lightrag/index.py --set llm.num_ctx=16384 --set lightrag.entity_extraction_use_json=true
```

Precedence: `config.yaml` < `-m profile` < `--set` < dedicated flags. Useful extras:
`--check` (test both models and exit), `--print-config` (show the resolved settings),
`--limit N` (index only N documents), `--log-level DEBUG`. Values like
`${OPENAI_API_KEY:-not-needed}` are expanded from the environment.

### `llm:` keys

| Key | Default | Notes |
|---|---|---|
| `provider`, `model` | `ollama`, `qwen2.5:7b-instruct` | |
| `base_url`, `api_key` | provider default | openai: `http://localhost:8000/v1`; ollama: `http://localhost:11434` (or `OLLAMA_HOST`) |
| `temperature`, `top_p`, `seed`, `stop` | `0.0`, -, -, - | |
| `max_tokens` | `4096` | max new tokens per call |
| `concurrency` | `4` | parallel requests (also fed to the framework's own concurrency setting) |
| `timeout`, `max_retries` | `600`, `3` | retries cover connection errors, timeouts, 429/5xx |
| `structured_output` | `auto` | JSON requests: `auto` tries JSON-schema, then JSON mode, then prompt-only |
| `strip_thinking` | `true` | drop `<think>...</think>` from reasoning models |
| `num_ctx`, `think`, `keep_alive`, `options` | `32768`, -, -, `{}` | **ollama** only - keep `num_ctx` large, Ollama's default is tiny |
| `extra_body`, `extra_headers` | `{}` | **openai** only, e.g. `{chat_template_kwargs: {enable_thinking: false}}` for vLLM |
| `device`, `dtype`, `quantization`, `trust_remote_code` | `auto`, `auto`, -, `false` | **huggingface** only |
| `batch_size`, `batch_wait_ms` | `1`, `50` | **huggingface**: dynamic batching of concurrent requests |
| `model_kwargs`, `chat_template_kwargs` | `{}` | **huggingface**: passed to `from_pretrained` / `apply_chat_template` |

### `embedding:` keys

| Key | Default | Notes |
|---|---|---|
| `provider`, `model`, `base_url`, `api_key` | `ollama`, `bge-m3` | |
| `dimensions` | auto | detected with a probe request when unset |
| `batch_size`, `concurrency`, `max_tokens` | `32`, `4`, `8192` | |
| `normalize` | `true` | L2-normalise vectors |
| `query_prefix`, `document_prefix` | `""` | for asymmetric models (e5: `query: ` / `passage: `, nomic: `search_query: ` / `search_document: `) |
| `query_instruction_template` | `Instruct: {instruction}\nQuery: ` | used when a method supplies a task instruction (HippoRAG 2 queries) |
| `num_ctx`, `truncate`, `keep_alive`, `options` | -, `true` | **ollama** only |
| `send_dimensions` | `false` | forward `dimensions` to the API (Matryoshka models) |
| `device`, `dtype`, `hf_backend`, `pooling`, `model_kwargs` | `auto`, `auto`, `auto`, `mean` | **huggingface** only |

### `input:` keys

`path` (file or directory), `patterns` (globs), `recursive`, `text_field` / `title_field` / `id_field`
(for `.json`, `.jsonl`, `.csv` records - HippoRAG-style `[{"title", "text"}]` corpora work as-is),
`encoding`, `limit`. Text files become one document each, titled by file name; document ids are
content hashes, so unchanged documents are recognised on re-runs.

## Testing without models

`tests/smoke_test.py` starts a mock server that speaks both the OpenAI and the Ollama API
(answering each framework's extraction prompts in the expected format) and runs every
`index.py` against it:

```bash
python tests/smoke_test.py --python ms_graphrag=.venvs/graphrag/bin/python --python lightrag=.venvs/lightrag/bin/python --python hipporag2=.venvs/hipporag/bin/python
```

`tests/make_tiny_hf_models.py` writes tiny random Hugging Face checkpoints to exercise the
`huggingface` provider offline.

## Tips for local models

* **Context window.** GraphRAG community reports and LightRAG extraction prompts are long; give the
  LLM 16-32k tokens of context (`llm.num_ctx` for Ollama, `--max-model-len` for vLLM).
* **Model size.** Graph extraction quality drops sharply below ~7B parameters. LightRAG recommends
  32B-class models; 7-14B instruct models work for experiments.
* **Reasoning models** (Qwen3, DeepSeek-R1): disable thinking for extraction (`llm.think: false` with
  Ollama, `extra_body.chat_template_kwargs.enable_thinking: false` with vLLM,
  `chat_template_kwargs.enable_thinking: false` with Hugging Face). `<think>` blocks are stripped either way.
* **Throughput.** For more than a handful of documents serve the model with vLLM/SGLang
  (`configs/vllm.yaml`) and raise `llm.concurrency`; in-process `huggingface` generation is the slowest option
  (use `llm.batch_size` > 1 to batch concurrent requests).
* **Offline machines.** tiktoken downloads its encodings on first use. If `litellm` is installed (GraphRAG
  pulls it in), its bundled `cl100k_base` / `o200k_base` files are used automatically; otherwise set
  `TIKTOKEN_CACHE_DIR` to a directory with pre-downloaded encodings. Hugging Face models can be given as
  local paths (`HF_HUB_OFFLINE=1`).
