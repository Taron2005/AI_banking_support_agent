from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..utils.logging import setup_logging
from .factory import build_runtime_orchestrator
from .llm_config import load_llm_settings
from .orchestrator import RuntimeRequest
from .runtime_config import load_runtime_settings
from .session_state import SessionStateStore


def run_chat(
    project_root: str,
    config_path: str | None,
    runtime_config_path: str | None,
    llm_config_path: str | None,
    index_name: str,
    verbose: bool,
) -> None:
    cfg = load_config(project_root=project_root, config_yaml=config_path)
    runtime_settings = load_runtime_settings(runtime_config_path)
    llm_settings = load_llm_settings(llm_config_path)
    orchestrator = build_runtime_orchestrator(
        app_config=cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
    sessions = SessionStateStore()
    state = sessions.get_or_create("local-cli")
    print("Runtime chat started. Type `exit` to quit.")
    while True:
        q = input("You> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        resp = orchestrator.handle(
            RuntimeRequest(session_id=state.session_id, query=q, index_name=index_name, verbose=verbose),
            state,
        )
        print("Assistant>")
        safe_answer = resp.answer_text.encode("ascii", errors="backslashreplace").decode("ascii")
        safe_obj = json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)
        safe_obj = safe_obj.encode("ascii", errors="backslashreplace").decode("ascii")
        print(safe_answer)
        print(safe_obj)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Runtime text chat")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--index-name", type=str, required=True)
    p.add_argument("--runtime-config", type=str, default=None)
    p.add_argument("--llm-config", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    run_chat(
        args.project_root,
        args.config,
        args.runtime_config,
        args.llm_config,
        args.index_name,
        args.verbose,
    )


if __name__ == "__main__":
    main()

