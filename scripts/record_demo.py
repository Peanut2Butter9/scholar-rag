"""Record real answers from a running ScholarRAG server for the static demo.

Start the server first (from the project root), then run this script:
    RATE_LIMIT_PER_MINUTE=100 .venv/bin/uvicorn app:app --app-dir src --port 8000
    .venv/bin/python scripts/record_demo.py

The responses are saved unchanged to deploy/demo_recordings.json; scripts/build_space.py --static replays them.
"""
import json
import time
import urllib.request
from datetime import date
from pathlib import Path

API = "http://localhost:8000/api/ask"
OUT = Path(__file__).resolve().parent.parent / "deploy" / "demo_recordings.json"

# (label, pertanyaan, broaden): campuran jawaban, parafrase, penolakan, dan pertanyaan luas
QUESTIONS = [
    ("Specific question", "Which two benchmark datasets are used to validate the Iterative Refinement Process (IRP)?", False),
    ("Paraphrased question", "Is the footage in the UBnormal benchmark recorded by real cameras, or is it made some other way?", False),
    ("Paraphrased question", "What auxiliary loss is minimized alongside the Real-NVP loss for visual anomaly detection on mobile robots?", False),
    ("Specific question", "What are the four main components of the Prior Normality Prompt Transformer (PNPT) framework?", False),
    ("Specific question", "What is the name of the open-source deep learning library for unsupervised anomaly detection and localization that supports OpenVINO?", False),
    ("Specific question", "Which methods are used for unsupervised anomaly detection in medical images such as brain MRI?", False),
    ("Near-miss: paper in corpus, detail not", "Which CUDA and PyTorch versions were used when measuring the inference time of TMUAD?", False),
    ("Near-miss: paper in corpus, detail not", "What was the monetary cost in USD of the GPT-4V API requests during the zero-shot evaluation of LogSAD?", False),
    ("Paper outside the corpus", "What image-level AUROC scores did the Denoising Diffusion Anomaly Detection (DDAD) framework achieve on MVTec and VisA?", False),
    ("Broad question", "I am interested to study about the unsupervised medical anomaly detection, what are their current research?", False),
    ("Broad question, Broaden search on", "I am interested to study about the unsupervised medical anomaly detection, what are their current research?", True),
]


def ask(question, broaden):
    body = json.dumps({"question": question, "broaden": broaden}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


if __name__ == "__main__":
    recs = []
    for label, q, broaden in QUESTIONS:
        d = ask(q, broaden)
        recs.append({"label": label, "question": q, "broaden": broaden, "response": d})
        print(f"{'answered' if d['answerable'] else 'refused ':8s} {d['timings_s']['total']:5.1f}s  {label}: {q[:70]}")
        time.sleep(5)   # jeda supaya tidak menabrak batas per menit Gemini
    OUT.write_text(json.dumps({"recorded": date.today().isoformat(), "items": recs}, ensure_ascii=False, indent=1))
    print(f"disimpan: {OUT} ({len(recs)} rekaman)")
