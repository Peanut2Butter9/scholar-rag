"""ScholarRAG pipeline: retrieval (BM25 ∪ dense → bge-reranker) and answers with citations.

Same settings and prompt as 03_hybrid_rerank.ipynb (final setup) and 04_answers_citations.ipynb
(PROMPT_VERSION v1), so results are comparable. Used by 05_evaluation.ipynb and, later, the API (M6).
"""
import json, re, time, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

BASE = Path.home() / "Projects" / "scholar-rag" / "data"
QP = "Represent this sentence for searching relevant passages: "   # prefix query khusus model bge


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"-\s+(\w)", r"\1", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def tok(s):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in ENGLISH_STOP_WORDS]


def load_chunks():
    ch = pd.read_json(BASE / "chunks_clean.jsonl", lines=True, dtype={"paper_id": str})
    ch["norm"] = ch.text.map(norm)
    return ch


class Retriever:
    """BM25 top-k ∪ dense top-k, reranked by bge-reranker-base; returns the top passages."""

    def __init__(self, ch, pool_k=10, top_k=5):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from rank_bm25 import BM25Okapi
        self.ch, self.pool_k, self.top_k = ch, pool_k, top_k
        assert json.load(open(BASE / "index" / "chunk_ids.json")) == ch.chunk_id.tolist(), "cache embedding tidak cocok"
        self.E = np.load(BASE / "index" / "emb_bge_small.npy")
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.bm25 = BM25Okapi([tok(t) for t in ch.text])
        self.ce = CrossEncoder("BAAI/bge-reranker-base", max_length=512)

    def retrieve(self, question):
        """Return (chunk positions, rerank scores, seconds per stage)."""
        t = {}
        t0 = time.perf_counter()
        qv = self.emb.encode([QP + question], normalize_embeddings=True)[0]
        dense = np.argsort(-(self.E @ qv))[:self.pool_k].tolist()
        t["dense"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sparse = np.argsort(-self.bm25.get_scores(tok(question)))[:self.pool_k].tolist()
        t["bm25"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pool = list(dict.fromkeys(sparse + dense))
        scores = self.ce.predict([(question, self.ch.text.iat[i]) for i in pool], batch_size=16, show_progress_bar=False)
        order = np.argsort(-scores)[:self.top_k]
        t["rerank"] = time.perf_counter() - t0
        return [pool[j] for j in order], [float(scores[j]) for j in order], t


# --- jawaban dengan sitasi (sama persis dengan notebook 04, PROMPT_VERSION v1) ---

ANSWER_MODEL = "gemini-3.5-flash-lite"
PROMPT_VERSION = "v1"


class Citation(BaseModel):
    passage: int = Field(description="Number of the passage (1-5) the quote comes from")
    quote: str = Field(description="Exact words copied from that passage, 5-40 words, supporting the answer")


class Answer(BaseModel):
    answerable: bool = Field(description="True only if the passages contain the information asked for")
    answer: str = Field(description="1-3 sentence answer based only on the passages; empty if not answerable")
    citations: list[Citation] = Field(description="Every passage used for the answer; empty if not answerable")


SYSTEM = """You answer questions about research papers on visual anomaly detection, using ONLY the numbered passages you are given.

- If the passages contain the information the question asks for, set answerable to true, answer in 1-3 sentences, and cite every passage you used. Each citation gives the passage number and a quote copied word for word from that passage (5-40 words) that supports the answer.
- If the passages do not contain the specific information asked for, set answerable to false, and leave answer and citations empty.
- A passage about the same paper or topic that does not state the specific detail asked for does NOT make the question answerable. Do not guess, infer missing numbers, or use outside knowledge."""


def build_prompt(ch, question, idxs):
    parts = []
    for n, i in enumerate(idxs, start=1):
        r = ch.iloc[i]
        parts.append(f"[{n}] {r.title} (pages {r.page}-{r.page_end})\n{r.text}")
    return f"Question: {question}\n\nPassages:\n\n" + "\n\n".join(parts)


class DailyQuotaExceeded(RuntimeError):
    """Kuota harian free tier habis: mencoba ulang hari ini tidak ada gunanya."""


def call_with_retry(fn, tries=5, wait=20):
    """Ulangi panggilan API saat 429/503 (rate limit per menit atau model sedang penuh)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "PerDay" in msg:
                raise DailyQuotaExceeded(msg[:300]) from e
            if any(s in msg for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")) and attempt < tries - 1:
                time.sleep(wait * (attempt + 1))
                continue
            raise


def answer(client, ch, question, idxs, model=ANSWER_MODEL):
    from google.genai import types
    config = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0,
                                         response_mime_type="application/json", response_schema=Answer)
    t0 = time.perf_counter()
    r = call_with_retry(lambda: client.models.generate_content(model=model, contents=build_prompt(ch, question, idxs),
                                                               config=config))
    u = r.usage_metadata
    return {"model": model, "prompt_version": PROMPT_VERSION,
            "parsed": r.parsed.model_dump() if r.parsed is not None else None, "raw": r.text,
            "latency_s": round(time.perf_counter() - t0, 2),
            "tokens_in": u.prompt_token_count, "tokens_out": u.candidates_token_count}


def check_citations(ch, parsed, idxs):
    """Setiap sitasi harus menunjuk passage 1..k dan kutipannya ada persis di passage itu."""
    if not parsed or not parsed["answerable"]:
        return False
    cits = parsed["citations"]
    return bool(cits) and all(1 <= c["passage"] <= len(idxs) and norm(c["quote"]) in ch.norm.iat[idxs[c["passage"] - 1]]
                              for c in cits)
