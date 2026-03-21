from voice_ai_banking_support_agent.runtime.llm import GeminiRESTClient


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict:
        return self._payload


def test_gemini_client_parses_response(monkeypatch) -> None:
    def _fake_post(url, params, json, timeout):  # noqa: ANN001
        return _Resp(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "Պատասխան"}]}}
                ]
            }
        )

    monkeypatch.setattr("voice_ai_banking_support_agent.runtime.llm.requests.post", _fake_post)
    client = GeminiRESTClient(
        model="gemini-2.0-flash",
        api_key="k",
        timeout_seconds=10,
        temperature=0.1,
    )
    out = client.generate("prompt")
    assert out == "Պատասխան"

