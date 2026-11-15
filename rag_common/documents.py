"""Corpus loading and chunking shared by the indexing scripts.

Supported inputs (file or directory, filtered by glob ``patterns``):

* ``.md`` / ``.markdown`` / ``.txt`` / ``.rst`` - one document per file
* ``.jsonl`` / ``.ndjson`` - one JSON record per line
* ``.json`` - a list of records, ``{"data"|"documents"|"corpus": [...]}``, or a single record
* ``.csv`` / ``.tsv`` - one record per row

Records need a text field (``text`` by default; ``title`` and ``id`` are optional),
which also covers HippoRAG-style corpora (``[{"title": ..., "text": ...}]``).
"""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".rst"}
RECORD_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}


@dataclass
class Document:
    id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def content_id(text: str, prefix: str = "doc-") -> str:
    """Stable id derived from the content (same scheme LightRAG uses by default)."""
    return prefix + hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _iter_files(root: Path, patterns: list[str], recursive: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    candidates = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(candidates):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            if any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
                yield path


def _records_from_json(path: Path, encoding: str) -> list[dict[str, Any]]:
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding=encoding) as fh:
            return [json.loads(line) for line in fh if line.strip()]
    data = json.loads(path.read_text(encoding=encoding))
    if isinstance(data, dict):
        for key in ("data", "documents", "docs", "corpus", "records", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: unsupported JSON structure ({type(data).__name__})")


def _records_from_csv(path: Path, encoding: str) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def load_documents(
    path: str | Path,
    *,
    patterns: list[str] | None = None,
    recursive: bool = True,
    text_field: str = "text",
    title_field: str = "title",
    id_field: str = "id",
    encoding: str = "utf-8",
    limit: int | None = None,
) -> list[Document]:
    """Load documents from ``path`` (see module docstring for formats)."""
    root = Path(path)
    patterns = patterns or ["*"]
    docs: list[Document] = []
    seen: set[str] = set()
    base = root if root.is_dir() else root.parent

    def _add(doc: Document) -> bool:
        if not doc.text.strip():
            return False
        if doc.id in seen:
            logger.debug("skipping duplicate document %s (%s)", doc.id, doc.source)
            return False
        seen.add(doc.id)
        docs.append(doc)
        return limit is not None and len(docs) >= limit

    for file in _iter_files(root, patterns, recursive):
        suffix = file.suffix.lower()
        rel = str(file.relative_to(base))
        if suffix in TEXT_SUFFIXES:
            text = file.read_text(encoding=encoding, errors="replace")
            if _add(Document(id=content_id(text), text=text, title=file.stem, source=rel)):
                break
        elif suffix in RECORD_SUFFIXES:
            records = _records_from_csv(file, encoding) if suffix in (".csv", ".tsv") else _records_from_json(file, encoding)
            stop = False
            for i, rec in enumerate(records):
                if isinstance(rec, str):
                    rec = {text_field: rec}
                if not isinstance(rec, dict) or text_field not in rec:
                    raise ValueError(
                        f"{file}: record {i} has no '{text_field}' field "
                        f"(set input.text_field; available keys: {list(rec)[:10] if isinstance(rec, dict) else type(rec).__name__})"
                    )
                text = str(rec.get(text_field) or "")
                title = str(rec.get(title_field) or "")
                rid = rec.get(id_field)
                meta = {k: v for k, v in rec.items() if k not in (text_field, title_field, id_field)}
                doc = Document(
                    id=str(rid) if rid not in (None, "") else content_id(f"{title}\n{text}"),
                    text=text,
                    title=title,
                    source=f"{rel}#{i}",
                    metadata=meta,
                )
                if _add(doc):
                    stop = True
                    break
            if stop:
                break
        else:
            logger.debug("skipping unsupported file %s", file)

    if not docs:
        raise ValueError(f"No documents found in {root} (patterns={patterns})")
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    doc_id: str
    title: str
    index: int


def _token_codec(encoding_name: str):
    import tiktoken

    ensure_tiktoken_cache()
    enc = tiktoken.get_encoding(encoding_name)
    return (lambda s: enc.encode(s, disallowed_special=())), enc.decode


def chunk_documents(
    docs: list[Document],
    *,
    chunk_size: int | None,
    overlap: int = 0,
    by: str = "token",
    encoding_name: str = "cl100k_base",
    prepend_title: bool = True,
) -> list[Chunk]:
    """Split documents into overlapping windows of ``chunk_size`` tokens (or words).

    ``chunk_size`` of ``None``/``0`` keeps every document as a single chunk.
    """
    if overlap and chunk_size and overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    if chunk_size and by == "token":
        encode, decode = _token_codec(encoding_name)
    elif chunk_size and by == "word":
        encode, decode = (lambda s: s.split()), (lambda ws: " ".join(ws))
    elif chunk_size:
        raise ValueError("chunk_by must be 'token' or 'word'")

    chunks: list[Chunk] = []
    for doc in docs:
        text = doc.text.strip()
        pieces: list[str]
        if not chunk_size:
            pieces = [text]
        else:
            units = encode(text)
            step = chunk_size - (overlap or 0)
            starts = list(range(0, max(len(units), 1), step))
            # drop a trailing window that is fully contained in the previous one
            if len(starts) > 1 and len(units) - starts[-1] <= (overlap or 0):
                starts = starts[:-1]
            pieces = [decode(units[s : s + chunk_size]).strip() for s in starts]
            pieces = [p for p in pieces if p]
        for i, piece in enumerate(pieces):
            body = f"{doc.title}\n{piece}" if (prepend_title and doc.title) else piece
            chunks.append(Chunk(text=body, doc_id=doc.id, title=doc.title, index=i))
    return chunks


def ensure_tiktoken_cache() -> None:
    """Make tiktoken work offline when possible.

    tiktoken downloads its BPE files on first use. LiteLLM ships the
    ``cl100k_base`` and ``o200k_base`` files; if LiteLLM is installed and
    ``TIKTOKEN_CACHE_DIR`` is unset, point tiktoken at that bundled copy.
    """
    import importlib.util
    import os

    if os.environ.get("TIKTOKEN_CACHE_DIR"):
        return
    spec = importlib.util.find_spec("litellm")
    if spec is None or not spec.origin:
        return
    bundled = Path(spec.origin).parent / "litellm_core_utils" / "tokenizers"
    if bundled.is_dir():
        os.environ["TIKTOKEN_CACHE_DIR"] = str(bundled)
