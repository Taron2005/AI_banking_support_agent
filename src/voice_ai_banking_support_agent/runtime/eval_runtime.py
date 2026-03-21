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

EVAL_QUERIES = [
    "Ամերիաբանկը ինչ սպառողական վարկեր ունի",
    "Ի՞նչ ավանդներ կան ACBA-ում",
    "Որտե՞ղ է IDBank-ի մոտակա մասնաճյուղը",
    "What are the deposit options at Ameriabank?",
    "Which branches are in Gyumri?",
    "իսկ դոլարով?",
    "իսկ Ամերիայի դեպքում?",
    "Which bank is best for loans?",
    "What is the exchange rate?",
    "Tell me about cards",
]


def _safe_console_text(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Runtime smoke evaluation")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--index-name", type=str, required=True)
    p.add_argument("--runtime-config", type=str, default=None)
    p.add_argument("--llm-config", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(project_root=args.project_root, config_yaml=args.config)
    runtime_settings = load_runtime_settings(args.runtime_config)
    llm_settings = load_llm_settings(args.llm_config)
    orchestrator = build_runtime_orchestrator(
        app_config=cfg, runtime_settings=runtime_settings, llm_settings=llm_settings
    )
    sessions = SessionStateStore()
    state = sessions.get_or_create("eval")
    for q in EVAL_QUERIES:
        out = orchestrator.handle(
            RuntimeRequest(session_id="eval", query=q, index_name=args.index_name, verbose=args.verbose),
            state,
        )
        print(_safe_console_text(json.dumps({"query": q, **out.model_dump()}, ensure_ascii=False)))


if __name__ == "__main__":
    main()

