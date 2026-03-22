from voice_ai_banking_support_agent.indexing.embedder import resolve_embedding_device


def test_resolve_embedding_device_cpu_literal():
    assert resolve_embedding_device("cpu") == "cpu"


def test_resolve_embedding_device_unknown_falls_back():
    assert resolve_embedding_device("not-a-device") == "cpu"
