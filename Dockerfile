# HuggingFace Space (Docker SDK) — Armenian OCR webapp.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces convention: non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ARMENIAN_OCR_MODELS_SOURCE=hf
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
# CPU-only torch wheels (~200 MB instead of the CUDA build)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user armenian_ocr/ ./armenian_ocr/
COPY --chown=user app/ ./app/

EXPOSE 7860
# single worker only: the job store lives in process memory (app/jobs.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
