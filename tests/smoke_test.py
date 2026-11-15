#!/usr/bin/env python
"""Offline end-to-end smoke test for the indexing scripts.

Starts the mock LLM server (tests/mock_llm_server.py) and runs every selected
method's ``index.py`` against it with the ``openai`` and ``ollama`` providers,
then checks the exit code and the produced index files. No GPU, model download
or real LLM server is needed - it verifies that the installation and the glue
code work.

    python tests/smoke_test.py                                  # all methods, current interpreter
    python tests/smoke_test.py --methods lightrag --providers ollama
    python tests/smoke_test.py --python ms_graphrag=.venvs/graphrag/bin/python \\
                               --python lightrag=.venvs/lightrag/bin/python \\
                               --python hipporag2=.venvs/hipporag/bin/python
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from mock_llm_server import serve  # noqa: E402

METHODS = {
    # method folder -> files that must exist (relative to the output dir) after indexing
    "ms_graphrag": ["output/entities.parquet", "output/relationships.parquet", "output/community_reports.parquet"],
    "lightrag": ["graph_chunk_entity_relation.graphml", "vdb_entities.json", "kv_store_doc_status.json"],
    "hipporag2": ["*/graph.pickle", "*/chunk_embeddings/vdb_chunk.parquet"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--providers", nargs="+", default=["openai", "ollama"], choices=["openai", "ollama"])
    parser.add_argument("--python", action="append", default=[], metavar="METHOD=PATH",
                        help="interpreter to use for a method (e.g. its own virtualenv)")
    parser.add_argument("--port", type=int, default=11499)
    parser.add_argument("--keep", action="store_true", help="keep the temporary output directory")
    args = parser.parse_args()

    interpreters = dict(item.split("=", 1) for item in args.python)
    server = serve(args.port, dim=48)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = {"openai": f"http://127.0.0.1:{args.port}/v1", "ollama": f"http://127.0.0.1:{args.port}"}

    workdir = Path(tempfile.mkdtemp(prefix="graphrag-smoke-"))
    results: list[tuple[str, str, bool, float, str]] = []
    try:
        for method in args.methods:
            for provider in args.providers:
                out = workdir / f"{method}-{provider}"
                cmd = [
                    interpreters.get(method, sys.executable), str(REPO_ROOT / method / "index.py"),
                    "-i", str(REPO_ROOT / "data" / "sample"), "-o", str(out),
                    "--llm-provider", provider, "--llm-model", "mock", "--llm-base-url", base[provider],
                    "--embedding-provider", provider, "--embedding-model", "mock", "--embedding-base-url", base[provider],
                    "--set", "llm.max_retries=0",
                ]
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
                elapsed = time.perf_counter() - t0
                missing = [p for p in METHODS[method] if not list(out.glob(p))]
                ok = proc.returncode == 0 and not missing
                detail = "" if ok else (f"exit={proc.returncode} missing={missing}\n" + proc.stdout[-3000:] + proc.stderr[-3000:])
                results.append((method, provider, ok, elapsed, detail))
                print(f"[{'PASS' if ok else 'FAIL'}] {method:<12} {provider:<7} {elapsed:6.1f}s", flush=True)
    finally:
        server.shutdown()
        if args.keep:
            print(f"outputs kept in {workdir}")
        else:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

    failed = [r for r in results if not r[2]]
    for method, provider, _, _, detail in failed:
        print(f"\n===== {method} / {provider} =====\n{detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
