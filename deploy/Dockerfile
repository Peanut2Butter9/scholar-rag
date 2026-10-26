# Image untuk Hugging Face Spaces (SDK: Docker). Dirakit oleh scripts/build_space.py.
FROM python:3.12-slim

# Spaces menjalankan container sebagai user dengan uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SCHOLARRAG_DATA=/home/user/app/data
WORKDIR /home/user/app

# torch versi CPU lebih dulu (jauh lebih kecil dari versi CUDA), lalu sisanya
COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements-app.txt

# unduh model sekali saat build, supaya startup tidak bergantung pada Hugging Face Hub
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY --chown=user src ./src
COPY --chown=user data ./data

EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
