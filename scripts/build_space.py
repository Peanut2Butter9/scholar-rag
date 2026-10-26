"""Assemble the Hugging Face Space folder for ScholarRAG, and optionally upload it.

    .venv/bin/python scripts/build_space.py                       # builds dist/space/
    .venv/bin/python scripts/build_space.py --push USER/scholarrag  # builds, then uploads to that Space

Uploading needs a Hugging Face login on this machine first (`.venv/bin/hf auth login`, or
`huggingface-cli login` on older versions). The Gemini key is NOT uploaded: add it yourself as the Space
secret GEMINI_API_KEY (Settings → Variables and secrets).
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "space"

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


def push(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Update ScholarRAG app")
    print(f"diunggah ke https://huggingface.co/spaces/{repo_id}")
    print("Jangan lupa: tambahkan secret GEMINI_API_KEY di Settings → Variables and secrets.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--push", metavar="USER/SPACE", help="upload to this Hugging Face Space after building")
    args = ap.parse_args()
    build()
    if args.push:
        push(args.push)
