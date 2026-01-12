from agentic.rag.simple_retriever import SimpleRetriever


def test_simple_retriever_returns_chunks(tmp_path):
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "a.md").write_text("# Scratch\nMinor scratch info\n", encoding="utf-8")

    r = SimpleRetriever(knowledge_dir=kb)
    out = r.retrieve("scratch minor", top_k=3)
    assert out, "Expected non-empty retrieval"
    assert out[0].source == "a.md"
