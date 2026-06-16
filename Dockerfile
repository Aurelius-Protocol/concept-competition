# syntax=docker/dockerfile:1-labs
# Self-contained scorer image: baked Gemma 3 weights + frozen prompt pool.
#
# The prequant step needs a GPU at *build* time (bitsandbytes quantizes on CUDA), exposed
# via BuildKit's CDI device support (labs syntax above; needs nvidia-ctk CDI spec on host).
#
# Build (google/gemma-3-12b-it is gated -> pass an HF token as a build secret; pin the SHA):
#   docker build \
#     --allow device \
#     --secret id=hf_token,env=HF_TOKEN \
#     --build-arg HF_REVISION=<40-char-sha> \
#     -t concept-scorer .
#
# Run (needs a CUDA GPU with ~12 GB VRAM for the NF4 12B):
#   docker run --gpus all -p 8000:8000 concept-scorer
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# Ubuntu 22.04's own `python3.11` package is a frozen 3.11.0 RC1, so pull a real
# 3.11.x from the deadsnakes PPA instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv && \
    python3.11 -m venv /opt/venv && \
    rm -rf /var/lib/apt/lists/*

# All later `python`/`pip` calls (incl. the runtime stage, which is FROM base) resolve to
# this venv, so the installer and the interpreter that runs our scripts always agree.
# (The venv has its own pip, sidestepping Debian's disabled-ensurepip/system-pip rules.)
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt requirements-build.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt -r requirements-build.txt

# ---- builder: bake model + freeze prompt pool ----
FROM base AS builder
ARG HF_REVISION
COPY . /app
# Pre-quantize the 12B to NF4 so only ~7-8 GB is baked (vs ~24 GB bf16).
# --device exposes the host GPU to this build step (CDI); requires `--allow device`.
# ldconfig: BuildKit injects the CDI driver libs but skips the ld-cache hook, so refresh
# it ourselves or libcuda/libnvidia-ml stay unresolvable (torch then sees no GPU).
RUN --mount=type=secret,id=hf_token --device=nvidia.com/gpu=all \
    ldconfig && \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python scripts/download_model.py --mode prequant ${HF_REVISION:+--revision $HF_REVISION}
# Freeze the held-out alpaca-cleaned pool into /app/data.
RUN python scripts/build_freeze_pool.py --pool-size 20000

# ---- runtime ----
FROM base AS runtime
# Never reach the network at runtime; the model + pool are baked in.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Propagate the pinned SHA to the *runtime* so HF_REVISION is the single source of truth:
# it picks the baked checkpoint (builder stage) AND becomes the revision the running scorer
# reports in /info, /healthz, and every score fingerprint. ARG does not cross stage
# boundaries, so re-declare it here. Empty when the build-arg is omitted -> the scorer falls
# back to config/competition.yaml's revision (which download_model.py already forced to be a
# real, non-placeholder SHA at build time, so the reported value is never the placeholder).
ARG HF_REVISION
ENV CONCEPT_SCORER_MODEL_REVISION=${HF_REVISION}
COPY --from=builder /opt/models /opt/models
COPY --from=builder /app/data /app/data
COPY --from=builder /app/concept_scorer /app/concept_scorer
COPY --from=builder /app/config /app/config
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
RUN python -m pip install --no-cache-dir --no-deps -e /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/readyz').status==200 else 1)" || exit 1

CMD ["uvicorn", "concept_scorer.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
