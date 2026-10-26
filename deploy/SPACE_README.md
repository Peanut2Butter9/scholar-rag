---
title: ScholarRAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers from 200 anomaly-detection papers
---

# ScholarRAG

Ask questions about 200 arXiv papers on visual anomaly detection. Each answer cites the passage it comes from, with the quote checked against the source, or the app says the papers don't contain the answer.

Retrieval: BM25 ∪ dense (`bge-small-en-v1.5`), reranked by `bge-reranker-base`. Answers: Gemini (`gemini-3.5-flash-lite`). The evaluation results are in the app's "About" panel.

This Space needs the secret `GEMINI_API_KEY` (Settings → Variables and secrets).
