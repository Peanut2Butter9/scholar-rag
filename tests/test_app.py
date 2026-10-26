"""Tests for the API (src/app.py) with a fake retriever and fake Gemini calls: no models, no API quota."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import app as app_module   # noqa: E402
import rag                 # noqa: E402

TEXTS = [
    "PatchCore builds a memory bank of patch features from normal images and scores test patches by distance. " * 3,
    "The UBnormal benchmark is generated using the Cinema4D software with virtual animated characters.",
    "We train for 20 epochs with the Adam optimizer and a learning rate of 0.0001 on one GPU.",
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, question, search_query=None, rerank_with="question"):
        self.calls.append((question, search_query, rerank_with))
        return [0, 1, 2], [0.9, 0.8, 0.7], {"dense": 0.01, "bm25": 0.01, "rerank": 0.1}


def fake_answer(answerable=True, quote="builds a memory bank of patch features from normal images", passage=1):
    def _answer(client, ch, question, idxs, model=rag.ANSWER_MODEL, tries=5):
        cits = [{"passage": passage, "quote": quote}] if answerable else []
        return {"model": model, "prompt_version": rag.PROMPT_VERSION, "latency_s": 0.5, "raw": "{}",
                "tokens_in": 10, "tokens_out": 5,
                "parsed": {"answerable": answerable, "answer": "A memory bank of patch features." if answerable else "",
                           "citations": cits}}
    return _answer


@pytest.fixture
def client(monkeypatch):
    ch = pd.DataFrame({"chunk_id": ["p1_c0000", "p2_c0000", "p3_c0000"], "paper_id": ["2106.08265v2", "2111.08644v3", "cs_0101001"],
                       "title": ["PatchCore", "UBnormal", "Old paper"], "page": [1, 2, 3], "page_end": [1, 2, 4], "text": TEXTS})
    ch["norm"] = ch.text.map(rag.norm)
    retriever = FakeRetriever()
    monkeypatch.setattr(app_module, "state", {"ch": ch, "retriever": retriever, "client": object()})
    monkeypatch.setattr(app_module, "limiter", app_module.RateLimiter(per_minute=100, per_day=1000))
    monkeypatch.setattr(rag, "answer", fake_answer())
    monkeypatch.setattr(rag, "rewrite_query", lambda client, q, model=None, tries=5: ("patch memory bank search", 0.3))
    c = TestClient(app_module.app)   # tanpa "with": lifespan (pemuatan model) tidak dijalankan
    c.retriever = retriever
    return c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["chunks"] == 3


def test_answer_with_verified_citation(client):
    d = client.post("/api/ask", json={"question": "How does PatchCore score patches?"}).json()
    assert d["answerable"] and d["citations_valid"]
    assert d["citations"][0]["verified"] and d["passages"][0]["cited"]
    assert d["search_query"] is None
    assert d["passages"][2]["url"] == "https://arxiv.org/abs/cs/0101001" and d["passages"][2]["pages"] == "3-4"
    assert set(d["timings_s"]) >= {"dense", "bm25", "rerank", "llm", "total"}


def test_quote_not_in_passage_is_flagged(client, monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer(quote="a sentence that is nowhere in the passage"))
    d = client.post("/api/ask", json={"question": "How does PatchCore score patches?"}).json()
    assert d["answerable"] and not d["citations_valid"] and not d["citations"][0]["verified"]


def test_refusal(client, monkeypatch):
    monkeypatch.setattr(rag, "answer", fake_answer(answerable=False))
    d = client.post("/api/ask", json={"question": "How much GPU memory does it need?"}).json()
    assert not d["answerable"] and d["answer"] == "" and d["citations"] == [] and not d["citations_valid"]


def test_broaden_passes_search_query(client):
    d = client.post("/api/ask", json={"question": "tell me about patch methods", "broaden": True}).json()
    assert d["search_query"] == "patch memory bank search" and "rewrite" in d["timings_s"]
    assert client.retriever.calls[-1] == ("tell me about patch methods", "patch memory bank search", "search")


def test_without_broaden_uses_plain_retrieval(client):
    client.post("/api/ask", json={"question": "How does PatchCore score patches?"})
    assert client.retriever.calls[-1][1] is None


def test_input_validation(client):
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422


def test_rate_limit_per_visitor(client, monkeypatch):
    monkeypatch.setattr(app_module, "limiter", app_module.RateLimiter(per_minute=2, per_day=1000))
    codes = [client.post("/api/ask", json={"question": "How does PatchCore work?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_windows():
    rl = app_module.RateLimiter(per_minute=1, per_day=2)
    assert rl.check("a", now=0) is None
    assert rl.check("a", now=10) is not None          # sama pengunjung, masih dalam satu menit
    assert rl.check("b", now=10) is None               # pengunjung lain boleh
    assert "daily" in rl.check("c", now=20)            # batas harian server (2) tercapai
    assert rl.check("c", now=86401) is None            # 24 jam kemudian boleh lagi


def test_snippet_centres_on_quote():
    text = " ".join(f"w{i}" for i in range(300))
    out, truncated = app_module.snippet(text, ["w200 w201 w202"], n_words=40)
    assert truncated and "w200 w201 w202" in out and out.startswith("… ") and out.endswith(" …")
    assert len(out.replace("… ", "").replace(" …", "").split()) == 40
    assert app_module.snippet("short passage", [], n_words=40) == ("short passage", False)
    assert app_module.snippet(text, [], n_words=0) == (text, False)
