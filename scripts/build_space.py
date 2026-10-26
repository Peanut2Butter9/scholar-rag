"""Assemble a Hugging Face Space for ScholarRAG, and optionally upload it.

Live app (Docker Space; needs a PRO account on Hugging Face):
    .venv/bin/python scripts/build_space.py                         # builds dist/space/
    .venv/bin/python scripts/build_space.py --push USER/scholarrag  # builds, then uploads

Recorded demo (static Space; free): replays deploy/demo_recordings.json (made by scripts/record_demo.py)
in the same web UI, without a server or model calls.
    .venv/bin/python scripts/build_space.py --static --push USER/scholarrag

Uploading needs a Hugging Face login on this machine first (`.venv/bin/hf auth login`). For the live app the
Gemini key is NOT uploaded: add it yourself as the Space secret GEMINI_API_KEY (Settings → Variables and secrets).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "space"
OUT_STATIC = ROOT / "dist" / "static-space"

STATIC_README = """---
title: ScholarRAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
short_description: Recorded demo of cited answers from 200 AD papers
---

# ScholarRAG (recorded demo)

Retrieval-augmented question answering over 200 arXiv papers on visual anomaly detection. Each answer cites the passage it comes from, with the quote checked against the source, or the system says the papers don't contain the answer.

This Space replays answers recorded from the live system (FastAPI + `bge-small-en-v1.5` / BM25 retrieval, `bge-reranker-base`, Gemini `gemini-3.5-flash-lite`); it makes no model calls. The "About" panel on the page lists the evaluation results.

Code, notebooks and the full evaluation: https://github.com/Peanut2Butter9/scholar-rag
"""

# hanya file yang dibutuhkan server; PDF dan file eksperimen tidak ikut
FILES = {
    "deploy/Dockerfile": "Dockerfile",
    "deploy/SPACE_README.md": "README.md",
    "requirements-app.txt": "requirements-app.txt",
    "src/app.py": "src/app.py",
    "src/rag.py": "src/rag.py",
    "src/static/index.html": "src/static/index.html",
    "data/chunks_clean.jsonl": "data/chunks_clean.jsonl",
    "data/index/emb_bge_small.npy": "data/index/emb_bge_small.npy",
    "data/index/chunk_ids.json": "data/index/chunk_ids.json",
}


def build():
    missing = [src for src in FILES if not (ROOT / src).exists()]
    if missing:
        raise SystemExit(f"File tidak ditemukan: {missing}")
    # cache embedding harus cocok dengan chunk yang dikirim
    ids = [json.loads(line)["chunk_id"] for line in open(ROOT / "data/chunks_clean.jsonl")]
    if json.load(open(ROOT / "data/index/chunk_ids.json")) != ids:
        raise SystemExit("chunk_ids.json tidak cocok dengan chunks_clean.jsonl; jalankan ulang notebook yang membuat embedding")
    if OUT.exists():
        shutil.rmtree(OUT)
    for src, dst in FILES.items():
        (OUT / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, OUT / dst)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"dist/space siap: {len(FILES)} file, {size / 1e6:.1f} MB")


def build_static():
    rec = json.load(open(ROOT / "deploy" / "demo_recordings.json"))
    html = (ROOT / "src" / "static" / "index.html").read_text()
    data = json.dumps(rec, ensure_ascii=False).replace("</", "<\\/")   # aman di dalam <script>
    marker = "<script>\n"
    assert html.count(marker) == 1, "index.html harus punya tepat satu <script>"
    html = html.replace(marker, f"<script>window.SCHOLARRAG_DEMO = {data};</script>\n" + marker)
    if OUT_STATIC.exists():
        shutil.rmtree(OUT_STATIC)
    OUT_STATIC.mkdir(parents=True)
    (OUT_STATIC / "index.html").write_text(html)
    (OUT_STATIC / "README.md").write_text(STATIC_README)
    print(f"dist/static-space siap: {len(rec['items'])} rekaman dari {rec['recorded']}, "
          f"{(OUT_STATIC / 'index.html').stat().st_size / 1e3:.0f} kB")


def push(repo_id, folder, sdk):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk=sdk, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="space",
                      commit_message="Update ScholarRAG " + ("recorded demo" if sdk == "static" else "app"))
    print(f"diunggah ke https://huggingface.co/spaces/{repo_id}")
    if sdk == "docker":
        print("Jangan lupa: tambahkan secret GEMINI_API_KEY di Settings → Variables and secrets.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--push", metavar="USER/SPACE", help="upload to this Hugging Face Space after building")
    ap.add_argument("--static", action="store_true", help="build the free recorded demo instead of the live app")
    args = ap.parse_args()
    if args.static:
        build_static()
    else:
        build()
    if args.push:
        push(args.push, OUT_STATIC if args.static else OUT, "static" if args.static else "docker")
