# concept-scorer

A **self-contained Docker scoring module** for the Apex Steering Competition. It runs
inside a validator but knows nothing about Bittensor — it only evaluates and scores
concept-steering submissions against a pinned **Gemma 4 31B** model and returns
`score = hit_rate`.

> **Model note:** the competition spec originally named `gemma-3-12b-it`. The model has
> been changed to **Gemma 4**, which ships no 12B variant, so this module pins
> **`google/gemma-4-31B-it`** (dense, `hidden_size=5376`, 60 layers). The steering
> `direction` is therefore shape **`(5376,)`**; layer 32 remains the fixed steer layer.

## What it does

For a submission and the active weekly concept, it:

1. validates the safetensors submission (`direction` `(5376,)` float32 unit-norm + metadata
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
concept-scorer smoke    --floor 0.5                                       # weather reference (GPU)
```

## Build & run

```bash
# Gemma 4 is gated: pass an HF token as a build secret and pin the revision SHA.
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg HF_REVISION=<40-char-sha> \
  -t concept-scorer .

docker run --gpus all -p 8000:8000 concept-scorer   # needs >=24 GB VRAM
```

The build bakes the model (pre-quantized to NF4, ~18–20 GB) and freezes the held-out
`unsloth/alpaca-cleaned` prompt pool into the image. At runtime the container never
touches the network (`HF_HUB_OFFLINE=1`).

## Configuration

`config/competition.yaml` is the single source of pinned values: model repo/revision SHA,
quantization (NF4/bf16), submission rules (shape/dtype/norm tolerance/alpha bounds),
generation params (greedy, seed, `max_new_tokens`), prompt-pool params, allowed concepts,
and detector versions. **Fill in the placeholder `revision` and `dataset_revision` SHAs
before building.** Library versions are pinned in `requirements.txt` (reproducibility).

## Pre-launch artifacts

- **Prompt pool** — frozen at build by `scripts/build_freeze_pool.py`.
- **Weather reference vector** — `scripts/build_weather_reference.py` derives the known-good
  `(5376,)` steering vector (diff-of-means at layer 32) on the real model; run once and
  commit/bake `concept_scorer/weather/reference_direction.safetensors` for the smoke test.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not gpu"     # detectors, submission validation, prompt sampling, steering hook, API
pytest -m gpu           # weather smoke test (requires CUDA + baked model)
```

The non-GPU suite needs only CPU torch (for the steering-hook unit test) plus
fastapi/pydantic; submission validation, config, detectors, and prompt sampling are
pure-stdlib.
