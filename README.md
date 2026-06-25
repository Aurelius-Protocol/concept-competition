# concept-scorer

A **self-contained Docker scoring module** for the Apex Steering Competition. It runs
inside a validator but knows nothing about Bittensor — it only evaluates and scores
concept-steering submissions against a pinned **Gemma 3 12B** model and returns a per-concept
`score` (aggregated as `hit_rate` or `graded`, per the `scoring` config).

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
4. runs the concept's pinned weighted-lexicon detector and returns the day-score — `hit_rate`
   (fraction of completions with intensity ≥ threshold, §8) or `graded` (mean normalized
   intensity in [0,1]).

The four competition concepts: `birthday_cake`, `medical_disclaimer`, `positive_sentiment`,
`hedging`. All four are **weighted lexicons**: a completion's raw concept-score is the
sum of matched cue weights. `birthday_cake`/`medical_disclaimer`/`hedging` use pinned
weighted-regex tables (**v2**); `positive_sentiment` uses the **AFINN-111** sentiment lexicon
(**v3**, net valence). Each concept's day-score is set by its `scoring` config — `hit_rate` (fraction of
completions whose intensity ≥ `threshold`, spec §8) or `graded` (mean normalized intensity in
`[0,1]`). **`graded` is a deliberate, config-gated deviation from §8** (`score = hit_rate`). The
weight tables are pinned in the detector classes (versioned); only `mode`/`threshold`/`saturation`
are config. All sit behind a `Detector` interface, so a detector can be swapped and its pinned
version bumped without touching callers. AFINN-111 is bundled under the ODbL; see
`concept_scorer/detectors/data/AFINN_NOTICE.txt`.

## Scope boundary

This module is **only** the per-submission scorer. Several parts of the competition spec are
deliberately **owned by the parent Bittensor validator**, not implemented here:

- **Weekly concept schedule (§4)** — which concept is active on a given day. The scorer is
  concept-agnostic: the caller passes `active_concept`, and a submission whose metadata
  `concept` doesn't match is rejected. It does not derive the active concept from the date.
- **Incentive / winner-take-all / decay (§10)** — leaderboard, leader tracking, emission
  decay, and hiding winners until a weekly round closes.
- **Apex query protocol & rate-limits (§7, §11c)** — pulling submissions from miner
  endpoints and enforcing the one-per-day-per-concept / four-per-day-per-hotkey caps.

The scorer evaluates a submission on receipt and returns the per-concept `score`; the above wraps it.

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
concept-scorer smoke    --floor 0.15                                      # weather reference (GPU)
```

## Getting started — operating the scorer

> **Canonical scoring requires CUDA + bitsandbytes NF4.** That is the only configuration whose
> scores count (SPEC §2). The Apple-Silicon / CPU path runs the model **unquantized in bf16** and is
> for **pipeline verification only** — every score it emits is self-labelled `"quantized": false` so
> non-canonical runs are obvious. Pick your path:
>
> - **Score real submissions (production):** CUDA, below.
> - **Verify the pipeline on a Mac first:** [Run locally on Apple Silicon](#run-locally-on-apple-silicon-mps).

### 0. Prerequisites (one-time)

- **Model access.** `google/gemma-3-12b-it` is **gated**: accept the license at
  huggingface.co/google/gemma-3-12b-it and have an HF token ready.
- **Pin the revision.** Replace the `REPLACE_WITH_PINNED_40_CHAR_SHA` placeholder in
  `config/competition.yaml` with the model's 40-char commit SHA. Both the image build and the model
  load **fail fast** on the placeholder — by design, so an unpinned model never loads.

### 1. Run the scorer service (CUDA + NF4)

Production is the warm HTTP service. Build the image (bakes the NF4 model + frozen prompt pool) and
run it:

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  --build-arg HF_REVISION=<40-char-sha> \
  -t concept-scorer .

docker run --gpus all -p 8000:8000 concept-scorer   # NF4 needs ~12 GB VRAM
```

The build pre-quantizes the model to NF4 (~7–8 GB) and freezes the held-out `unsloth/alpaca-cleaned`
prompt pool into the image; at runtime the container never touches the network (`HF_HUB_OFFLINE=1`).
Wait for readiness before sending work:

```bash
curl -sf localhost:8000/readyz     # 200 only once the model is warm (poll this)
curl -s  localhost:8000/healthz
# {"status":"ok","ready":true,"model_loaded":true,"model_revision":"<sha>","module_version":"0.1.0"}
```

> **Bare-metal CUDA (no Docker).** `pip install -r requirements.txt && pip install -e .`, point
> `CONCEPT_SCORER_MODEL_PATH` / `CONCEPT_SCORER_POOL_PATH` at the weights and frozen pool (see
> [Pre-launch artifacts](#pre-launch-artifacts)), then run the same entrypoint the Dockerfile uses:
> `uvicorn concept_scorer.api.app:create_app --factory --host 0.0.0.0 --port 8000`. Device is
> auto-detected, so on a CUDA box this loads NF4 with no extra flags.

### 2. Score a submission

A submission is a safetensors file with one `direction` `(3840,)` float32 unit-norm tensor and
metadata `alpha` / `layer` / `concept` (produced by miners). The **caller** supplies the
`active_concept`, `day_index`, and `seed` for the evaluation.

**HTTP (production):**

```bash
curl -s -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "active_concept": "positive_sentiment",
  "day_index": 0,
  "seed": 1234,
  "submission_path": "/path/to/sub.safetensors",
  "return_completions": false
}'
```

`submission_path` reads a file the container can see; use `submission_b64` to inline the bytes, or
`POST /score-file` for a multipart upload. A successful response (canonical CUDA shown):

```json
{"score":0.328,"hit_count":3,"total":150,"active_concept":"positive_sentiment","day_index":0,
 "seed":1234,"detector_version":"v3","model_revision":"<sha>","device":"cuda","quantized":true,
 "scoring_mode":"graded","alpha":8000.0,"completions":null,
 "timings_ms":{"sample":2.6,"generate":...,"detect":4.2}}
```

**CLI (one-shot; loads the model for the run):**

```bash
# cheap no-GPU pre-check — run before spending a GPU on a malformed file
concept-scorer validate --submission sub.safetensors --concept positive_sentiment
# {"error_code": "ok", "alpha": 8000.0, "layer": 32, "concept": "positive_sentiment"}

concept-scorer score --submission sub.safetensors --concept positive_sentiment --day-index 0 --seed 1234
# -> the same ScoreResponse payload as POST /score
```

### 3. Read the result

- **`score`** is the per-concept day-score: `hit_rate` (fraction of completions with intensity ≥
  threshold) or `graded` (mean normalized intensity in `[0,1]`), per `scoring_mode`.
- **Optional concentration penalty.** When a concept sets `sparsity_lambda > 0` in
  `config/competition.yaml`, the day-score is multiplied by `clamp(1 - sparsity_lambda·(1 - H), 0, 1)`,
  where `H` is the direction's Hoyer sparsity (`0` = dense/uniform, `1` = a single active dim) — this
  rewards concentrated, interpretable directions over diffuse brute-force ones, and keeps the score in
  `[0,1]`. It is **off by default** (`sparsity_lambda: 0.0`); to enable, raise the value, e.g.
  `hedging: {mode: graded, threshold: 2.0, saturation: 4.0, sparsity_lambda: 0.5}`. The result's
  `diagnostics` always reports `sparsity` (H), `raw_score`, and `sparsity_factor`, so you can calibrate
  `sparsity_lambda` against the real H distribution before turning it on.
- An **invalid** submission is a *rejection, not a zero*: HTTP **422** with a typed `error_code`
  (e.g. `not_unit_norm`, `bad_layer`, `concept_mismatch`). The CLI exits non-zero; pass
  `--reject-as-zero` to instead emit `{"score": 0.0, "error_code": ...}`.
- **`device` / `quantized` tell you whether the score is canonical.** Canonical is
  `"device":"cuda","quantized":true`; anything else (e.g. `"mps"`/`false`) is a verification run and
  must never be recorded as a real score.

## Run locally on Apple Silicon (MPS)

> **⚠️ Dev-only / not reproducible.** bitsandbytes NF4 is CUDA-only, so on MPS the model runs
> **unquantized bf16**. 4-bit dequant is not bit-identical to bf16, so steering directions and
> `alpha` calibrated on MPS will **not** reproduce on the pinned CUDA/NF4 validator (§2). Use
> MPS for plumbing and iteration only; do all canonical calibration and scoring on CUDA+NF4.
> The scorer logs a warning at load and tags every score and `/info` with `device` +
> `quantized`, so non-NF4 results are self-evident.

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
| `CONCEPT_SCORER_BACKEND` | `local` (in-process, steers), `vllm` (CUDA high-throughput, steers), or `openai` (LM Studio; baseline-only) |
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
CONCEPT_SCORER_MAX_PROMPTS=8 concept-scorer smoke --floor 0.15   # quick green first
concept-scorer smoke --floor 0.15                               # full 150-prompt smoke
```

Once smoke passes, exercise the **same operator surface as production** (§ Getting started) against
the real model to confirm the full validate → score → serve cycle — the only difference from CUDA is
that the score is bf16 and self-labelled non-canonical. With the env overlay above still exported
(add `CONCEPT_SCORER_MAX_PROMPTS=8` for a fast pass):

```bash
# one-shot CLI score (loads the warm bf16 model on MPS)
concept-scorer score --submission sub.safetensors --concept positive_sentiment --day-index 0 --seed 1234
# {"score":0.328,"hit_count":3,"total":8,"diagnostics":{... "device":"mps","quantized":false ...}}

# or the warm HTTP service — identical entrypoint to the Dockerfile, device auto-detected to MPS
uvicorn concept_scorer.api.app:create_app --factory --host 127.0.0.1 --port 8000
curl -sf localhost:8000/readyz && \
curl -s -X POST localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"active_concept":"positive_sentiment","day_index":0,"seed":1234,
       "submission_path":"sub.safetensors","return_completions":false}'
# {... "device":"mps","quantized":false ...}  <- the quantized:false flag marks this dev-only
```

This exact cycle was run end-to-end on `google/gemma-3-12b-it` on an Apple-Silicon Mac (MPS, bf16):
`resolve_layers` finds the 48 multimodal decoder layers and steers layer 32, and the `quantized:false`
flag keeps the result from being mistaken for a canonical score.

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
run it (start the LM Studio server and load a model first). Real (steered) scoring must use a
**white-box** backend — `local`, or `vllm` on CUDA — which has the forward-hook access steering needs.

## High-throughput CUDA backend (vLLM)

For high-throughput scoring on CUDA, the **vLLM backend** (`CONCEPT_SCORER_BACKEND=vllm`) runs an
in-process vLLM engine and applies the *same* layer-32 residual steering as the `local` backend, via
vLLM's continuous batching. It is **CUDA-only** (vLLM has no MPS/CPU path) and pins its own torch
stack, so install it in a **separate env** from `requirements.txt`:

```bash
pip install -r requirements-vllm.txt && pip install bitsandbytes   # vLLM brings its own torch
export CONCEPT_SCORER_BACKEND=vllm
export CONCEPT_SCORER_VLLM_QUANTIZATION=bitsandbytes   # NF4 — a bf16 12B won't fit a 24 GB GPU
export CONCEPT_SCORER_VLLM_MAX_MODEL_LEN=4096          # cap KV-cache reservation on small-VRAM GPUs
concept-scorer smoke --floor 0                         # "did it run"; vLLM scores are non-canonical
```

**Not canonical.** vLLM's engine/kernels are not bit-identical to the pinned transformers+NF4 path,
so its scores need a re-pin + re-baseline before they count (SPEC §2). On the same prompts it
reproduces the canonical NF4 hit-rate to within ~1/150 — a sanity check, not a re-baseline.

Tuning knobs (all CUDA-only; defaults in `config.py`):

| env var | meaning (default) |
| --- | --- |
| `CONCEPT_SCORER_VLLM_QUANTIZATION` | `None` (bf16) \| `bitsandbytes` \| `awq` \| `gptq` (default `None`) |
| `CONCEPT_SCORER_VLLM_DTYPE` | compute dtype (default `bfloat16`) |
| `CONCEPT_SCORER_VLLM_ENFORCE_EAGER` | keep the Python steering hook live; no CUDA graphs (default `1`) |
| `CONCEPT_SCORER_VLLM_MAX_MODEL_LEN` | context cap for KV-cache sizing; `None` = model max (default `None`) |
| `CONCEPT_SCORER_VLLM_GPU_MEM` | GPU memory utilization target 0–1 (default `0.90`) |
| `CONCEPT_SCORER_VLLM_MAX_NUM_SEQS` | max concurrent sequences (default `256`) |

## Configuration

`config/competition.yaml` is the single source of pinned values: model repo/revision SHA,
quantization (NF4/bf16), submission rules (shape/dtype/norm tolerance/alpha bounds),
generation params (greedy, seed, `max_new_tokens`), prompt-pool params, allowed concepts,
detector versions, and the per-concept `scoring` policy (`mode` `hit_rate`|`graded`,
`threshold`, `saturation`). **Fill in the placeholder `revision` SHA before building** — both the
build (`download_model.py`) and the model load **fail fast** on the
`REPLACE_WITH_PINNED_40_CHAR_SHA` placeholder rather than fetch an unpinned model.
(`dataset_revision` only affects rebuilding the already-frozen, sha256-checked pool.) Library
versions are pinned in `requirements.txt`; `requirements-mac.txt` is a **dev-only**,
non-reproducible overlay (newer torch/transformers, no bitsandbytes).

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
