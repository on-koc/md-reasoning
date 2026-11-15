"""Command-line + YAML settings handling shared by the indexing scripts.

Precedence (lowest -> highest): ``config.yaml`` < ``--set key=value`` < dedicated flags
(``--llm-model`` etc.). Relative paths inside the YAML file are resolved against the
file's directory; paths given on the command line are resolved against the CWD.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ConfigError,
    EmbeddingConfig,
    InputConfig,
    LLMConfig,
    expand_env,
    redact,
    resolve_path,
    set_by_path,
)

logger = logging.getLogger(__name__)

_PATH_KEYS = ("input.path", "output_dir")


@dataclass
class Settings:
    config_path: Path | None
    raw: dict[str, Any]
    llm: LLMConfig
    embedding: EmbeddingConfig
    input: InputConfig
    input_path: Path
    output_dir: Path
    method: dict[str, Any]
    """The method-specific section (``graphrag`` / ``lightrag`` / ``hipporag``)."""

    def dump(self) -> dict[str, Any]:
        return redact(
            {
                "config_path": str(self.config_path) if self.config_path else None,
                "input": {**asdict(self.input), "path": str(self.input_path)},
                "output_dir": str(self.output_dir),
                "llm": asdict(self.llm),
                "embedding": asdict(self.embedding),
                "method": self.method,
            }
        )


def build_parser(description: str, default_config: Path | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", type=Path, default=default_config, help="YAML config file (default: %(default)s)")
    p.add_argument("-i", "--input", help="input file or directory (overrides input.path)")
    p.add_argument("-o", "--output", help="output / working directory (overrides output_dir)")
    p.add_argument("--limit", type=int, help="only index the first N documents")

    g = p.add_argument_group("model overrides")
    g.add_argument(
        "-m",
        "--models",
        type=Path,
        help="YAML file with llm:/embedding: sections that replace the ones in --config "
        "(ready-made profiles live in configs/, e.g. configs/vllm.yaml)",
    )
    for kind in ("llm", "embedding"):
        g.add_argument(f"--{kind}-provider", help="huggingface | openai | ollama")
        g.add_argument(f"--{kind}-model", help="model id / name / path")
        g.add_argument(f"--{kind}-base-url", help="API base URL (openai / ollama providers)")
        g.add_argument(f"--{kind}-api-key", help="API key (if the server needs one)")
    g.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set llm.num_ctx=16384 --set graphrag.method=fast",
    )

    p.add_argument("--check", action="store_true", help="test the LLM + embedding connection and exit")
    p.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ... (default: %(default)s)")
    return p


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if logging.getLogger().level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "openai", "urllib3", "LiteLLM", "litellm"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
    key, value = item.split("=", 1)
    return key.strip(), yaml.safe_load(value) if value.strip() else None


def load_settings(args: argparse.Namespace, method_section: str) -> Settings:
    """Merge YAML file + CLI arguments into a validated :class:`Settings`."""
    cwd = Path.cwd()
    raw: dict[str, Any] = {}
    config_path: Path | None = None
    base_dir = cwd
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        base_dir = config_path.parent
    raw = expand_env(raw)

    # YAML paths are relative to the config file
    raw.setdefault("input", {})
    if raw["input"].get("path"):
        raw["input"]["path"] = str(resolve_path(raw["input"]["path"], base_dir))
    raw["output_dir"] = str(resolve_path(raw.get("output_dir") or "./output", base_dir))

    # --models profile: replaces the llm / embedding sections
    if getattr(args, "models", None):
        models_path = Path(args.models).expanduser().resolve()
        if not models_path.is_file():
            raise ConfigError(f"Models file not found: {models_path}")
        profile = expand_env(yaml.safe_load(models_path.read_text(encoding="utf-8")) or {})
        unknown = sorted(set(profile) - {"llm", "embedding"})
        if unknown:
            raise ConfigError(f"{models_path}: only 'llm' and 'embedding' sections are allowed, found {unknown}")
        for section in ("llm", "embedding"):
            if section in profile:
                raw[section] = profile[section] or {}

    # --set overrides (paths relative to CWD)
    for item in args.overrides:
        key, value = _parse_override(item)
        if key in _PATH_KEYS and value:
            value = str(resolve_path(str(value), cwd))
        set_by_path(raw, key, value)

    # dedicated flags win
    flag_map = {
        "input": "input.path",
        "output": "output_dir",
        "limit": "input.limit",
        "llm_provider": "llm.provider",
        "llm_model": "llm.model",
        "llm_base_url": "llm.base_url",
        "llm_api_key": "llm.api_key",
        "embedding_provider": "embedding.provider",
        "embedding_model": "embedding.model",
        "embedding_base_url": "embedding.base_url",
        "embedding_api_key": "embedding.api_key",
    }
    for attr, key in flag_map.items():
        value = getattr(args, attr, None)
        if value is None:
            continue
        if key in _PATH_KEYS:
            value = str(resolve_path(value, cwd))
        set_by_path(raw, key, value)

    llm = LLMConfig.from_dict(raw.get("llm"))
    embedding = EmbeddingConfig.from_dict(raw.get("embedding"))
    input_cfg = InputConfig.from_dict(raw.get("input"))
    input_path = resolve_path(input_cfg.path, base_dir)
    return Settings(
        config_path=config_path,
        raw=raw,
        llm=llm,
        embedding=embedding,
        input=input_cfg,
        input_path=input_path,
        output_dir=Path(raw["output_dir"]),
        method=copy.deepcopy(raw.get(method_section) or {}),
    )


def check_models(settings: Settings) -> bool:
    """Send one chat request, one JSON request and one embedding request."""
    from .embeddings import create_embedder
    from .llm import create_llm

    ok = True
    print(f"LLM       : {settings.llm.provider} / {settings.llm.model}")
    try:
        llm = create_llm(settings.llm)
        t0 = time.perf_counter()
        res = llm.generate("Reply with exactly one word: OK", max_tokens=16)
        print(f"  chat    : {res.text!r} (finish={res.finish_reason}, {time.perf_counter() - t0:.2f}s)")
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
            "required": ["city", "country"],
        }
        t0 = time.perf_counter()
        res = llm.generate("Name the capital city of France and its country as JSON.", json_schema=schema, max_tokens=128)
        try:
            parsed = json.loads(res.text)
            print(f"  json    : {parsed} ({time.perf_counter() - t0:.2f}s)")
        except json.JSONDecodeError:
            print(f"  json    : WARNING - output is not valid JSON: {res.text[:120]!r}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ERROR   : {type(exc).__name__}: {exc}")
    print(f"Embedding : {settings.embedding.provider} / {settings.embedding.model}")
    try:
        emb = create_embedder(settings.embedding)
        t0 = time.perf_counter()
        vecs = emb.embed(["hello world", "graph based retrieval"])
        print(f"  vectors : shape={tuple(vecs.shape)} ({time.perf_counter() - t0:.2f}s)")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ERROR   : {type(exc).__name__}: {exc}")
    return ok


def run(main) -> None:
    """Run an index script's ``main()`` and turn configuration / input errors
    into a one-line message instead of a traceback."""
    try:
        code = main()
    except (ConfigError, FileNotFoundError) as exc:
        logging.getLogger("rag_common").error("%s", exc)
        code = 2
    except KeyboardInterrupt:
        logging.getLogger("rag_common").warning("interrupted - re-run the same command to resume")
        code = 130
    sys.exit(code)


def handle_common_actions(args: argparse.Namespace, settings: Settings) -> None:
    """Implement ``--print-config`` and ``--check`` (both exit the process)."""
    if args.print_config:
        print(yaml.safe_dump(settings.dump(), sort_keys=False, allow_unicode=True))
        sys.exit(0)
    if args.check:
        sys.exit(0 if check_models(settings) else 1)


def log_usage(llm: Any, embedder: Any) -> None:
    s = llm.stats
    logger.info(
        "LLM usage (%s): %d requests, %d prompt tokens, %d completion tokens, %d truncated",
        llm.name, s["requests"], s["prompt_tokens"], s["completion_tokens"], s["truncated"],
    )
    logger.info("Embedding usage (%s): %d texts in %d calls", embedder.name, embedder.stats["texts"], embedder.stats["requests"])
