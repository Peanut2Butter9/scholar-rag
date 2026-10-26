"""ScholarRAG API (M6): question → retrieval → answer with verified citations, or a refusal.

Run from the project root:
    .venv/bin/uvicorn app:app --app-dir src --port 8000
then open http://localhost:8000. Startup loads the embedding model, BM25 index and reranker (~20 s).

Settings (environment variables):
    GEMINI_API_KEY          required
    SCHOLARRAG_DATA         data folder (default: data/ in the project root)
    RATE_LIMIT_PER_MINUTE   questions per minute per visitor (default 10)
    DAILY_QUESTION_LIMIT    questions per rolling 24 h for the whole server (default 200; protects the Gemini quota)
    SNIPPET_WORDS           words of each passage returned to the browser (default 80; 0 = whole passage)
    ALLOW_BROADEN           1 to allow the opt-in query rewriting ("broaden search"), 0 to disable (default 1)
    ACCESS_CODE             if set, /api/ask requires this code in the X-Access-Code header (used by scripts/share_live.sh)
"""
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import rag

SRC = Path(__file__).resolve().parent
load_dotenv(SRC.parent / ".env")

RATE_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
DAILY_LIMIT = int(os.getenv("DAILY_QUESTION_LIMIT", "200"))
SNIPPET_WORDS = int(os.getenv("SNIPPET_WORDS", "80"))
ALLOW_BROADEN = os.getenv("ALLOW_BROADEN", "1") == "1"
ACCESS_CODE = os.getenv("ACCESS_CODE", "")   # kosong = tanpa kode akses

state = {}
lock = threading.Lock()   # model torch dipakai satu request sekaligus


class RateLimiter:
    """Batas per pengunjung per menit dan batas harian untuk seluruh server (jendela bergulir, di memori)."""

    def __init__(self, per_minute, per_day):
        self.per_minute, self.per_day = per_minute, per_day
        self.visitors = defaultdict(deque)
        self.all = deque()
        self._lock = threading.Lock()

    def check(self, key, now=None):
        """Catat satu pertanyaan; kembalikan pesan error kalau batas terlewati, None kalau boleh."""
        now = time.time() if now is None else now
        with self._lock:
            mine = self.visitors[key]
            while mine and now - mine[0] > 60:
                mine.popleft()
            while self.all and now - self.all[0] > 86400:
                self.all.popleft()
            if len(self.all) >= self.per_day:
                return "This demo has reached its daily question limit. Please try again tomorrow."
            if len(mine) >= self.per_minute:
                return f"Too many questions in a minute (limit {self.per_minute}). Please wait a moment."
            mine.append(now)
            self.all.append(now)
            return None


limiter = RateLimiter(RATE_PER_MIN, DAILY_LIMIT)


@asynccontextmanager
async def lifespan(app):
    from google import genai
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY kosong; isi file .env atau secret server terlebih dahulu")
    state["ch"] = rag.load_chunks()
    state["retriever"] = rag.Retriever(state["ch"])
    state["client"] = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    yield
    state.clear()


app = FastAPI(title="ScholarRAG", version="0.2", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    broaden: bool = False   # rewriting kueri sebelum retrieval (opsional, lebih lambat)


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
    text: str               # potongan passage (SNIPPET_WORDS kata di sekitar kutipan pertama)
    truncated: bool
    rerank_score: float
    cited: bool


class AskResponse(BaseModel):
    question: str
    search_query: str | None   # kueri hasil rewriting, kalau broaden dipakai
    answerable: bool
    answer: str
    citations: list[CitationOut]
    citations_valid: bool
    passages: list[PassageOut]
    timings_s: dict[str, float]
    model: str


def arxiv_url(paper_id):
    return "https://arxiv.org/abs/" + paper_id.replace("_", "/", 1)


def client_key(request):
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def snippet(text, quotes, n_words=SNIPPET_WORDS):
    """Potong passage menjadi n_words kata, dimulai sedikit sebelum kutipan pertama yang ditemukan."""
    words = text.split()
    if n_words <= 0 or len(words) <= n_words:
        return text, False
    low = [w.lower() for w in words]
    start = 0
    for q in quotes:
        head = q.lower().split()[:3]
        hit = next((i for i in range(len(low) - len(head) + 1) if head and low[i:i + len(head)] == head), None)
        if hit is not None:
            start = hit
            break
    begin = max(0, min(start - n_words // 4, len(words) - n_words))
    end = begin + n_words
    out = " ".join(words[begin:end])
    return ("… " if begin > 0 else "") + out + (" …" if end < len(words) else ""), True


@app.get("/api/health")
def health():
    return {"status": "ok" if state else "starting", "chunks": len(state.get("ch", [])),
            "answer_model": rag.ANSWER_MODEL, "prompt_version": rag.PROMPT_VERSION,
            "rate_limit_per_minute": RATE_PER_MIN, "daily_question_limit": DAILY_LIMIT,
            "broaden_available": ALLOW_BROADEN, "access_code_required": bool(ACCESS_CODE)}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    msg = limiter.check(client_key(request))
    if msg:
        raise HTTPException(429, msg)
    if ACCESS_CODE and not hmac.compare_digest(request.headers.get("x-access-code", ""), ACCESS_CODE):
        raise HTTPException(401, "This demo needs the access code that came with the link.")
    ch, retriever, client = state["ch"], state["retriever"], state["client"]
    question = req.question.strip()
    t0 = time.perf_counter()
    timings = {}
    search_query = None
    try:
        if req.broaden and ALLOW_BROADEN:
            search_query, timings["rewrite"] = rag.rewrite_query(client, question, tries=2)
        with lock:
            idxs, scores, t = retriever.retrieve(question, search_query=search_query, rerank_with="search")
        a = rag.answer(client, ch, question, idxs, tries=2)
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
    passages = []
    for n, (i, s) in enumerate(zip(idxs, scores), start=1):
        r = ch.iloc[i]
        text, truncated = snippet(r.text, [c.quote for c in cit_out if c.passage == n])
        passages.append(PassageOut(passage=n, chunk_id=r.chunk_id, paper_id=r.paper_id, title=r.title,
                                   pages=f"{r.page}-{r.page_end}" if r.page != r.page_end else str(r.page),
                                   url=arxiv_url(r.paper_id), text=text, truncated=truncated,
                                   rerank_score=round(s, 3), cited=any(c.passage == n for c in cit_out)))
    timings.update({k: round(v, 2) for k, v in t.items()})
    timings["llm"] = a["latency_s"]
    timings["total"] = round(time.perf_counter() - t0, 2)
    return AskResponse(question=question, search_query=search_query, answerable=p["answerable"],
                       answer=p["answer"] if p["answerable"] else "", citations=cit_out,
                       citations_valid=bool(cit_out) and all(c.verified for c in cit_out),
                       passages=passages, timings_s=timings, model=a["model"])


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(SRC / "static" / "index.html")
