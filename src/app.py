"""ScholarRAG API (M6): question → retrieval → answer with verified citations, or a refusal.

Run from the project root:
    .venv/bin/uvicorn app:app --app-dir src --port 8000
then open http://localhost:8000. Startup loads the embedding model, BM25 index and reranker (~20 s).
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import rag

SRC = Path(__file__).resolve().parent
load_dotenv(SRC.parent / ".env")

state = {}
lock = threading.Lock()   # model torch (MPS) dipakai satu request sekaligus


@asynccontextmanager
async def lifespan(app):
    from google import genai
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY kosong; isi file .env terlebih dahulu")
    state["ch"] = rag.load_chunks()
    state["retriever"] = rag.Retriever(state["ch"])
    state["client"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    yield
    state.clear()


app = FastAPI(title="ScholarRAG", version="0.1", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class CitationOut(BaseModel):
    passage: int
    quote: str
    verified: bool          # kutipan ada persis di passage yang disebut


class PassageOut(BaseModel):
    passage: int
    chunk_id: str
    paper_id: str
    title: str
    pages: str
    url: str
    text: str
    rerank_score: float
    cited: bool


class AskResponse(BaseModel):
    question: str
    answerable: bool
    answer: str
    citations: list[CitationOut]
    citations_valid: bool
    passages: list[PassageOut]
    timings_s: dict[str, float]
    model: str


def arxiv_url(paper_id):
    return "https://arxiv.org/abs/" + paper_id.replace("_", "/", 1)


@app.get("/api/health")
def health():
    return {"status": "ok" if state else "starting", "chunks": len(state.get("ch", [])),
            "answer_model": rag.ANSWER_MODEL, "prompt_version": rag.PROMPT_VERSION}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    ch, retriever = state["ch"], state["retriever"]
    question = req.question.strip()
    t0 = time.perf_counter()
    with lock:
        idxs, scores, t = retriever.retrieve(question)
    try:
        a = rag.answer(state["client"], ch, question, idxs, tries=2)
    except rag.DailyQuotaExceeded:
        raise HTTPException(503, "The Gemini free-tier quota for today is used up. Try again after it resets.")
    except Exception as e:
        raise HTTPException(502, f"The LLM call failed: {str(e)[:200]}")
    p = a["parsed"]
    if p is None:
        raise HTTPException(502, "The LLM did not return valid JSON.")

    cits = p["citations"] if p["answerable"] else []
    cit_out = [CitationOut(passage=c["passage"], quote=c["quote"],
                           verified=1 <= c["passage"] <= len(idxs) and rag.norm(c["quote"]) in ch.norm.iat[idxs[c["passage"] - 1]])
               for c in cits]
    cited = {c.passage for c in cit_out}
    passages = []
    for n, (i, s) in enumerate(zip(idxs, scores), start=1):
        r = ch.iloc[i]
        passages.append(PassageOut(passage=n, chunk_id=r.chunk_id, paper_id=r.paper_id, title=r.title,
                                   pages=f"{r.page}-{r.page_end}" if r.page != r.page_end else str(r.page),
                                   url=arxiv_url(r.paper_id), text=r.text, rerank_score=round(s, 3), cited=n in cited))
    timings = {k: round(v, 2) for k, v in t.items()}
    timings["llm"] = a["latency_s"]
    timings["total"] = round(time.perf_counter() - t0, 2)
    return AskResponse(question=question, answerable=p["answerable"], answer=p["answer"] if p["answerable"] else "",
                       citations=cit_out, citations_valid=bool(cit_out) and all(c.verified for c in cit_out),
                       passages=passages, timings_s=timings, model=a["model"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(SRC / "static" / "index.html")
