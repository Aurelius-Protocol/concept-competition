# Self-contained scorer image: baked Gemma 4 weights + frozen prompt pool.
#
# Build (model is gated -> pass an HF token as a build secret; pin the revision SHA):
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg HF_REVISION=<40-char-sha> \
#     -t concept-scorer .
#
# Run (needs a CUDA GPU with >=24 GB VRAM):
#   docker run --gpus all -p 8000:8000 concept-scorer
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-build.txt

# ---- builder: bake model + freeze prompt pool ----
FROM base AS builder
ARG HF_REVISION
COPY . /app
# Pre-quantize the 31B to NF4 so only ~18-20 GB is baked (vs ~62 GB bf16).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python scripts/download_model.py --mode prequant ${HF_REVISION:+--revision $HF_REVISION}
# Freeze the held-out alpaca-cleaned pool into /app/data.
RUN python scripts/build_freeze_pool.py --pool-size 20000

# ---- runtime ----
FROM base AS runtime
# Never reach the network at runtime; the model + pool are baked in.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY --from=builder /opt/models /opt/models
COPY --from=builder /app/data /app/data
COPY --from=builder /app/concept_scorer /app/concept_scorer
COPY --from=builder /app/config /app/config
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir --no-deps -e /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz').status==200 else 1)" || exit 1

CMD ["uvicorn", "concept_scorer.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
