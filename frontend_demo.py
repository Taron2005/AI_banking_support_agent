from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server

from voice_ai_banking_support_agent.config import load_config
from voice_ai_banking_support_agent.runtime.factory import build_runtime_orchestrator
from voice_ai_banking_support_agent.runtime.llm_config import load_llm_settings
from voice_ai_banking_support_agent.runtime.orchestrator import RuntimeRequest
from voice_ai_banking_support_agent.runtime.runtime_config import load_runtime_settings
from voice_ai_banking_support_agent.runtime.session_state import SessionStateStore


def _render_page(answer: str, trace: list[str], query: str, index_name: str) -> str:
    safe_answer = html.escape(answer)
    safe_query = html.escape(query)
    safe_index = html.escape(index_name)
    trace_html = "".join(f"<li>{html.escape(x)}</li>" for x in trace)
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Voice AI Banking Demo</title></head>
<body style="font-family: Arial; max-width: 900px; margin: 20px auto;">
  <h2>Banking Assistant Frontend Demo</h2>
  <p>Runtime-only demo (text). Uses your production runtime orchestrator.</p>
  <form method="POST">
    <label>Index name:</label><br/>
    <input name="index_name" value="{safe_index}" style="width:100%;"/><br/><br/>
    <label>Question:</label><br/>
    <textarea name="query" rows="4" style="width:100%;">{safe_query}</textarea><br/><br/>
    <button type="submit">Ask</button>
  </form>
  <hr/>
  <h3>Answer</h3>
  <pre style="white-space: pre-wrap;">{safe_answer}</pre>
  <h3>Decision trace</h3>
  <ul>{trace_html}</ul>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="One-file frontend demo for runtime QA")
    p.add_argument("--project-root", type=str, default=".")
    p.add_argument("--config", type=str, default="validation_manifest_update_multi.yaml")
    p.add_argument("--runtime-config", type=str, default="runtime_config.yaml")
    p.add_argument("--llm-config", type=str, default=None)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--index-name", type=str, default="multi_model_index")
    args = p.parse_args()

    project_root = Path(args.project_root).resolve()
    app_cfg = load_config(project_root=project_root, config_yaml=Path(args.config).resolve())
    runtime_cfg = load_runtime_settings(Path(args.runtime_config).resolve())
    llm_cfg = load_llm_settings(Path(args.llm_config).resolve() if args.llm_config else None)
    orchestrator = build_runtime_orchestrator(
        app_config=app_cfg, runtime_settings=runtime_cfg, llm_settings=llm_cfg
    )
    states = SessionStateStore()
    session_id = "frontend-demo-session"

    def app(environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET").upper()
        query = ""
        index_name = args.index_name
        answer = "Type a question and press Ask."
        trace: list[str] = []
        if method == "POST":
            size = int(environ.get("CONTENT_LENGTH", "0") or "0")
            body = environ["wsgi.input"].read(size).decode("utf-8", errors="ignore")
            form = parse_qs(body)
            query = (form.get("query", [""])[0]).strip()
            index_name = (form.get("index_name", [args.index_name])[0]).strip() or args.index_name
            state = states.get_or_create(session_id)
            out = orchestrator.handle(
                RuntimeRequest(
                    session_id=session_id,
                    query=query,
                    index_name=index_name,
                    verbose=True,
                ),
                state,
            )
            answer = out.answer_text + "\n\n" + json.dumps(
                {"status": out.status, "refusal_reason": out.refusal_reason, "detected_topic": out.detected_topic},
                ensure_ascii=False,
                indent=2,
            )
            trace = out.decision_trace

        data = _render_page(answer=answer, trace=trace, query=query, index_name=index_name).encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(data)))])
        return [data]

    with make_server(args.host, args.port, app) as httpd:
        print(f"Frontend running at http://{args.host}:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

