from __future__ import annotations

import os

# Before any Hugging Face / Google libs: avoid TensorFlow (often breaks on Windows with newer protobuf)
# and force pure-Python protobuf if TF or other stacks already mixed versions.
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse

import uvicorn

try:
    from pathlib import Path

    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parent
    load_dotenv(_root / ".env", override=False)
except ImportError:
    pass

from voice_ai_banking_support_agent.runtime.api import build_app


def main() -> None:
    p = argparse.ArgumentParser(description="Run runtime FastAPI server")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--config", type=str, default="validation_manifest_update_hy.yaml")
    p.add_argument("--runtime-config", type=str, default="runtime_config.yaml")
    p.add_argument("--llm-config", type=str, default="llm_config.yaml")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    app = build_app(
        project_root=args.project_root,
        config_path=args.config,
        runtime_config_path=args.runtime_config,
        llm_config_path=args.llm_config,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

