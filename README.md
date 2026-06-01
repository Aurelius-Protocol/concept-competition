# concept-scorer

A **self-contained Docker scoring module** for the Apex Steering Competition. It runs
inside a validator but knows nothing about Bittensor — it only evaluates and scores
concept-steering submissions against a pinned **Gemma 3 12B** model and returns
`score = hit_rate`.

> **Model note:** this module pins **`google/gemma-3-12b-it`** (`hidden_size=3840`, 48
> layers); the steering `direction` is shape **`(3840,)`** and layer 32 is the fixed steer
> layer. The model is **gated** on Hugging Face — accept the Gemma license and provide an
> HF token to download it.

## What it does

For a submission and the active weekly concept, it:

1. validates the safetensors submission (`direction` `(3840,)` float32 unit-norm + metadata
   `alpha`/`layer`/`concept`);
2. registers a forward hook that adds `alpha × direction` to the **layer-32 residual
   stream** at every token position;
3. greedily generates completions for that day's ~150 frozen prompts (deterministic from
   `(day_index, seed)`, never reused across days);
4. runs the concept's pinned detector and returns the **hit rate**.

The four competition concepts: `birthday_cake`, `medical_disclaimer`, `positive_sentiment`,
`hedging`. Detectors are version-pinned regex lexicons today, behind a `Detector` interface
so a pinned NLP classifier can be slotted in later (notably for `positive_sentiment`)
without touching callers.

## Interfaces

**HTTP service** (warm model, `uvicorn`):

- `POST /score` — JSON `{active_concept, day_index, seed, submission_b64|submission_path,
  return_completions}` → `ScoreResponse`. Invalid submissions return **HTTP 422** with a
  typed `error_code` (a rejection, not a score of 0).
- `POST /score-file` — same, via multipart file upload.
- `GET /readyz` — 200 only when the model is loaded & warm (poll this before sending work).
- `GET /healthz`, `GET /info` — liveness and the pinned config payload.

**CLI** (`concept-scorer`):

```
concept-scorer info
concept-scorer validate --submission sub.safetensors --concept hedging   # no GPU
concept-scorer score    --submission sub.safetensors --concept hedging --day-index 0 --seed 1234
concept-scorer smoke    --floor 0.25                                      # weather reference (GPU)
```

## Build & run

```bash
# google/gemma-3-12b-it is GATED — accept the Gemma license on HF and pass your token as a
# build secret. Pin the revision SHA for a reproducible build.
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg HF_REVISION=<40-char-sha> \
  -t concept-scorer .

docker run --gpus all -p 8000:8000 concept-scorer   # NF4 needs ~12 GB VRAM
```

The build bakes the model (pre-quantized to NF4, ~7–8 GB) and freezes the held-out
`unsloth/alpaca-cleaned` prompt pool into the image. At runtime the container never
touches the network (`HF_HUB_OFFLINE=1`).

## Run locally on Apple Silicon (MPS)

The Docker image above is **CUDA-only** (bitsandbytes 4-bit + `nvidia/cuda`); Docker
Desktop on macOS can't reach the Apple GPU. To test on an Apple-Silicon Mac, run
**bare-metal** instead. There is no bitsandbytes on Apple Silicon, so the model runs
**unquantized in bf16** on the Metal/MPS backend (the 12B is ~24 GB in bf16, well within a
128 GB machine and fine on smaller Macs too). The device is **auto-detected** — the same
code picks CUDA in the container and MPS here; nothing in the pinned `competition.yaml`,
`requirements.txt`, or `Dockerfile` changes.

```bash
# 0. environment (uv installs a managed Python 3.12; torch wheels prefer it over 3.14)
brew install uv
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-mac.txt && uv pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1   # CPU-fallback for any op MPS doesn't implement yet
```

Local runs are configured by an **environment overlay** (read in `config.py`; defaults
reproduce the CUDA behavior, so unset = no change):

| env var | meaning |
| --- | --- |
| `CONCEPT_SCORER_DEVICE` | `auto`\|`cuda`\|`mps`\|`cpu` (default `auto`) |
| `CONCEPT_SCORER_QUANTIZE` | `auto`\|`on`\|`off` (`auto` = 4-bit only on CUDA) |
| `CONCEPT_SCORER_MODEL_PATH` | load weights from a host dir (don't bake into an image) |
| `CONCEPT_SCORER_MODEL_REPO` | override the HF repo id (e.g. an ungated mirror of the pinned repo) |
| `CONCEPT_SCORER_MODEL_REVISION` | model revision/SHA (use `main` for a quick local run) |
| `CONCEPT_SCORER_POOL_PATH` | path to the frozen prompt pool on the host |
| `CONCEPT_SCORER_MAX_PROMPTS` | cap effective `per_day` (fast first smoke) |
| `CONCEPT_SCORER_ALPHA_MIN` / `_MAX` | override submission alpha bounds locally (steering-strength calibration) |
| `CONCEPT_SCORER_BACKEND` | `local` (in-process, steers) or `openai` (LM Studio; baseline-only) |
| `CONCEPT_SCORER_OPENAI_BASE_URL` / `_OPENAI_MODEL` / `_OPENAI_API_KEY` | endpoint, model id, and key for the `openai` backend |
| `CONCEPT_SCORER_ALLOW_UNSTEERED` | `1` to let the `openai` backend run an unsteered baseline (the API equivalent of CLI `--baseline`) |
| `CONCEPT_SCORER_CONFIG` | use an alternate config file (e.g. the dev overlay) |

**Fast plumbing dry-run on a tiny model** (~135 MB, a couple of minutes) — validates the
full pipeline incl. the steering hook before committing to the ~24 GB model:

```bash
export CONCEPT_SCORER_CONFIG=config/competition.dev.yaml
python scripts/build_freeze_pool.py --pool-size 256 --out data/dev_pool.jsonl
python scripts/build_weather_reference.py --out /tmp/dev_weather.safetensors
concept-scorer smoke --reference /tmp/dev_weather.safetensors --floor 0.0
unset CONCEPT_SCORER_CONFIG
```

**Real 12B run** — `google/gemma-3-12b-it` is **gated**: accept the license at
huggingface.co/google/gemma-3-12b-it, then `hf auth login` (stores your token), then:

```bash
export CONCEPT_SCORER_MODEL_PATH=$PWD/models/gemma-3-12b-it
export CONCEPT_SCORER_MODEL_REVISION=main
export CONCEPT_SCORER_POOL_PATH=$PWD/data/prompt_pool.jsonl

python scripts/download_model.py --mode snapshot          # ~24 GB bf16 weights -> MODEL_PATH
python scripts/build_freeze_pool.py --pool-size 20000     # frozen pool -> POOL_PATH
python scripts/build_weather_reference.py                 # weather vector on the real model (MPS)
CONCEPT_SCORER_MAX_PROMPTS=8 concept-scorer smoke --floor 0.25   # quick green first
concept-scorer smoke --floor 0.25                               # full 150-prompt smoke
```

Generation on MPS is memory-bandwidth bound, so a full 150-prompt smoke takes a while;
use `CONCEPT_SCORER_MAX_PROMPTS` for a fast first signal. If steering looks too weak
(steered ≈ unsteered), the `alpha` may need calibrating to the model's residual magnitude
— sweep it and, for local diagnostics, raise the bound with `CONCEPT_SCORER_ALPHA_MAX`.

## External backend (LM Studio): what works and what doesn't

You can point the scorer at any **OpenAI-compatible** server (e.g. LM Studio) with
`CONCEPT_SCORER_BACKEND=openai` + `CONCEPT_SCORER_OPENAI_BASE_URL=http://localhost:1234/v1`
(`host.docker.internal` from inside a container). **Important limitation:** that API is
**black-box** (text in / text out) — there is no way to inject a steering vector into the
residual stream. So this backend **cannot apply steering and cannot produce a valid
competition score.** It refuses a steered (`alpha != 0`) request:

```
$ CONCEPT_SCORER_BACKEND=openai CONCEPT_SCORER_OPENAI_BASE_URL=http://localhost:1234/v1 \
  concept-scorer score --submission sub.safetensors --concept hedging --day-index 0 --seed 1234
{"error_code": "steering_unsupported", "message": "... cannot apply residual steering ..."}
```

It is useful only for an **unsteered baseline** or plumbing checks — pass `--baseline` to
run it (start the LM Studio server and load a model first). Real (steered) scoring must use
the in-process `local` backend, which has the white-box forward-hook access steering needs.

## Configuration

`config/competition.yaml` is the single source of pinned values: model repo/revision SHA,
quantization (NF4/bf16), submission rules (shape/dtype/norm tolerance/alpha bounds),
generation params (greedy, seed, `max_new_tokens`), prompt-pool params, allowed concepts,
and detector versions. **Fill in the placeholder `revision` and `dataset_revision` SHAs
before building.** Library versions are pinned in `requirements.txt` (reproducibility).

## Pre-launch artifacts

- **Prompt pool** — frozen at build by `scripts/build_freeze_pool.py`.
- **Weather reference vector** — `scripts/build_weather_reference.py` derives the known-good
  `(3840,)` steering vector (diff-of-means at layer 32) on the real model; run once and
  commit/bake `concept_scorer/weather/reference_direction.safetensors` for the smoke test.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not gpu"     # detectors, submission validation, prompt sampling, steering hook, API
pytest -m gpu           # weather smoke test (requires a CUDA or Apple/MPS device + the model)
```

The non-GPU suite needs only CPU torch (for the steering-hook unit test) plus
fastapi/pydantic; submission validation, config, detectors, and prompt sampling are
pure-stdlib.
