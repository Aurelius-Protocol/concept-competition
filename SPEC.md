# Apex Steering Competition — Specification (as-built)

> Status: reflects the implementation in this repository as of 2026-06-02. This supersedes the
> original draft specification. Its structure mirrors the original (§1–§10 line up); a new §11
> (Scope boundary) is inserted, and the substantive differences are summarized at the end in
> [Changes from the original draft](#changes-from-the-original-draft).

## 1. Goal

Decentralized discovery of concept directions in Gemma-3-12B's residual stream that produce
reliable behavioral steering when added to layer-32 activations. This repository (`concept-scorer`)
is the **validator-side scoring module**: it validates a miner's steering submission, generates
completions under that steering, and returns a per-concept score. It runs inside a validator but
knows nothing about Bittensor — scheduling, incentive, and the miner-query protocol live one layer
up (see [§11, Scope boundary](#11-scope-boundary)).

Measurement is deterministic and lexicon-based: every concept is scored by a version-pinned
weighted lexicon, so hit rates are reproducible across validators running the identical model
configuration.

## 2. Model

`google/gemma-3-12b-it` at a pinned revision SHA. Quantization 4-bit NF4, bf16 compute. A single
canonical inference configuration; all miners calibrate against the identical setup.

The steering directions are specific to this exact configuration, so the following are pinned and
identical across all (canonical) validators: model revision SHA, quantization (NF4), compute dtype
(bf16), GPU architecture, and the `transformers` / `bitsandbytes` library versions. The 4-bit
dequantization kernels are not bit-identical across hardware or library versions, and pinning them
keeps hit rates reproducible.

- **Architecture (pinned, asserted at load):** `hidden_size = 3840`, `num_hidden_layers = 48`.
- **Revision pinning is enforced:** the config ships a `REPLACE_WITH_PINNED_40_CHAR_SHA`
  placeholder; both the image build (`scripts/download_model.py`) and the model load **fail fast**
  rather than fetch an unpinned revision from the hub. The operator fills the real SHA at build time.
- **Canonical backend is CUDA + bitsandbytes NF4.** An Apple-Silicon (MPS) / CPU path exists for
  **local development only**: bitsandbytes NF4 is CUDA-only, so off-CUDA the model runs
  **unquantized bf16**, which is *not* numerically identical to the canonical validator. That path
  logs a warning at load and every score is self-labeled with `device` and `quantized` so non-NF4
  results are identifiable and never mistaken for canonical ones. Calibration and scoring that
  count must run on CUDA + NF4.

### Throughput backend (vLLM) — CUDA verification (2026-06-02)

A high-throughput `vllm` backend (`CONCEPT_SCORER_BACKEND=vllm`) sits alongside the canonical
in-process `local` path and applies the **same** layer-32 residual-stream steering (forward hook)
under vLLM's continuous batching. It was verified end-to-end on CUDA (RTX 4090, 24 GB, WSL2).

- **Verified stack:** `vllm==0.22.0`, `torch==2.11.0+cu130`, `transformers==5.9.0`,
  `bitsandbytes==0.49.2`, against the non-gated `unsloth/gemma-3-12b-it-bnb-4bit` checkpoint with
  `CONCEPT_SCORER_VLLM_QUANTIZATION=bitsandbytes`.
- **Cross-backend agreement:** on the same 150 prompts (day 0, seed 1234) the vLLM-NF4 weather
  hit-rate was **28/150 = 0.187** vs the canonical transformers-NF4 **29/150 = 0.193** — within one
  hit. Steered-vs-`alpha=0` on vLLM was 3/16 vs 0/16, confirming the hook fires. This is a sanity
  check that the throughput path reproduces the steering behavior, **not** a re-baseline. This check
  is now automated as `tests/test_backend_parity_gpu.py` (`pytest -m gpu -k parity`): it scores the
  same reference submission through both backends and asserts the hit counts agree within ±2. The
  *structural* anti-divergence guard — both backends share `generation.encode_prompts` (identical
  tokens) and `steering.add_steering` (identical steering math) — runs without a GPU as
  `tests/test_backend_parity.py` in the CI no-GPU suite.
- **Still non-canonical for scoring:** vLLM's engine/kernels are not bit-identical to the pinned
  transformers+NF4 path, so vLLM scores need a re-pin + re-baseline before they count.
- **Driver-only-CUDA requirements (box without the CUDA toolkit / `nvcc`):** the backend forces
  vLLM's PyTorch-native sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`, set in `load()`) — FlashInfer's
  sampler JIT-needs `nvcc`, and greedy decode makes the native path argmax-identical. Triton's
  helper needs Python dev headers (`Python.h`), so build the env on a Python that ships them (e.g.
  a uv-managed CPython, not a headers-less system `python3.12`). On ≤24 GB GPUs cap the context via
  `CONCEPT_SCORER_VLLM_MAX_MODEL_LEN` (e.g. 4096): gemma-3-12b's 131072 window can't reserve KV
  cache beside the NF4 weights.

## 3. Layer

Residual stream at layer 32 — a single, fixed layer for the entire competition (matching the
open-source NLA reference work). Enforced at submission validation (`layer == 32`).

## 4. Concepts

Four concepts, increasing in difficulty:

| # | Concept | Detector (version) | Scoring mode | Threshold |
|---|---------|--------------------|--------------|-----------|
| 1 | Birthday-cake topic presence | weighted regex lexicon (`v3`) | `hit_rate` | 2.0 |
| 2 | Medical disclaimers | weighted regex lexicon (`v2`) | `hit_rate` | 2.0 |
| 3 | Positive sentiment | AFINN-111 lexicon (`v3`) | `graded` | 3.0 |
| 4 | Hedging language | weighted regex lexicon (`v3`) | `graded` | 2.0 |

**The scorer is concept-agnostic.** Each evaluation names an `active_concept`; the scorer routes
to that concept's pinned detector, prompt window, and scoring policy. A submission whose metadata
`concept` does not match the requested `active_concept` is rejected, and a request for a concept
outside the four allowed is rejected at the API boundary. The **weekly schedule** that decides which
concept is active on a given day (one per week, no carryover) is owned by the parent validator, not
this module ([§11](#11-scope-boundary)).

## 5. Detection

Each concept is scored by a **weighted lexicon**: a completion's raw concept-score is the sum of the
weights of the lexicon cues it matches, and it *hits* when that raw score is at least the concept's
pinned `threshold`. The cue/weight tables are pinned in the detector classes and versioned; only
`threshold` (and the scoring `mode`/`saturation`, §8) are config-tunable. Any NLP/lexicon component
is deterministic and version-pinned, so hit rates are reproducible.

- **Birthday-cake topic presence** (`v3`) — weighted regex over a birthday-cake vocabulary. Strong
  cues (`birthday cake`, `happy birthday`/`happy bday`, `birthday party/celebration`, `blow/light
  out the candles`, `birthday boy/girl`, `birthday wish`, `another year older`, `many happy returns`)
  carry weight ≥ threshold and hit on their own; generic trappings (`cake`, `cupcake`, `layer/tier
  cake`, `candles`, `frosting`, `icing`, `buttercream`, `fondant`, `sprinkles`, `party hat`,
  `birthday`, `bday`, `candlelight`) carry ~1.0, so two together hit while a lone incidental "cake"
  does not. Non-birthday "cake" contexts (`piece of cake` idiom, `wedding cake`) are vetoed.
- **Medical disclaimers** (`v2`) — weighted regex over disclaimer phrasings (advising consultation
  with a healthcare professional, "not a substitute for professional medical advice", "not medical
  advice", "for informational purposes only", "talk to your doctor", "if symptoms persist", …). Each
  phrasing is individually sufficient (weight = threshold).
- **Positive sentiment** (`v3`) — the **AFINN-111** sentiment lexicon (Finn Årup Nielsen; ~2,476
  words/phrases rated −5…+5), vendored verbatim and **sha256-pinned** (`detectors/data/afinn_111.txt`,
  ODbL). Scoring follows AFINN's intended use: it matches the lexicon's **multi-word phrase** entries
  (e.g. `not good` −2, `no fun` −3, `does not work` −3) and applies a **negation window** that flips
  the valence of a sentiment word following a negator (`not`, `no`, `never`, an `n't` contraction, …)
  within 2 tokens, reset by a contrastive cue (`but`, `however`). The net valence is the (possibly
  negated) sum, so negated positives ("not good", "isn't great") score negative rather than positive.
- **Hedging language** (`v3`) — weighted regex over a hedging-cue lexicon (`perhaps`, `possibly`,
  `maybe`, `might`, `may`, `likely`, `could`, `it seems`, `appears to`, `I think/believe/suppose/guess`,
  `in my opinion`, `sort of`, `kind of`, `arguably`, `presumably`, `I'm not sure/certain`, `it
  depends`, …); two distinct cues hit at the default threshold. Bare modals (`would`/`should`/`can`)
  are deliberately excluded.

## 6. Prompt set

`unsloth/alpaca-cleaned`. A held-out pool of **20,000** instruction prompts is assembled and frozen
before launch and **sha256-checked** on load. The validator samples **~150 prompts per day** from a
single seeded permutation of the pool: day *d* receives the disjoint contiguous window
`[d·150, (d+1)·150)`, so the selection is fully reproducible from `(day_index, seed)` and **prompts
are never reused across days**. Prompts are never revealed in advance.

## 7. Submission format

A single safetensors file containing one tensor:

- `direction`: shape `(3840,)`, dtype `float32`, L2-normalized to unit norm (within `1e-3`).
  Normalization places all steering strength in `alpha`, making submissions comparable across miners.

Required metadata:

- `alpha` (float) — steering strength (§8); must lie within the pinned bounds `[−32000, 32000]`
  (the calibrated range for gemma-3-12b's layer-32 residual magnitude — model/layer-specific).
- `layer` (int) — equals `32`.
- `concept` (str) — must match the active concept; a mismatch is rejected.

Each submission is ~15 KB. Validation is pure-Python (no GPU) and every failure is a typed rejection
(bad shape/dtype, non-unit-norm, non-finite values, wrong layer, concept mismatch, alpha out of
bounds, unknown concept) rather than a score of 0. The miner↔validator transport (pulling submissions
from miner endpoints, the per-hotkey daily caps) is owned by the parent validator ([§11](#11-scope-boundary)).

## 8. Scoring

Each concept's day-score is set by its pinned **scoring policy** — `{mode, threshold, saturation}` in
the `scoring` config block:

```
hit_rate:  score = (# completions with intensity ≥ threshold) / N           # spec-original
graded:    score = mean_i( clamp(intensity_i / saturation, 0, 1) )          # mean normalized intensity
```

- `intensity_i` is the detector's continuous per-completion score (summed cue weights / AFINN net
  valence). Under `graded`, a negative intensity floors at 0, and the day-score stays in `[0, 1]` —
  comparable to a `hit_rate` so the validator's incentive can rank either mode uniformly.
- **`graded` is a deliberate, config-gated deviation from the original "score = hit_rate".** It
  rewards *how strongly* a concept is present, not just whether it crossed the threshold. Per-concept
  defaults: birthday-cake and medical-disclaimer use `hit_rate` (presence concepts); positive-sentiment
  and hedging use `graded` (intensity/density concepts). Any concept can be flipped to `hit_rate` with
  a one-line config change.

**`alpha`** is the scalar steering strength: at evaluation the validator adds `alpha × direction` to
the layer-32 residual stream. With `direction` unit-normalized, `alpha` is the sole control of how
strongly the activation is pushed. Miners tune `alpha` against the pinned quantized model.

The weight tables (in code, versioned) and the `threshold`/`saturation` values (in config) are pinned
together; they are calibration knobs to be tuned against real steered gemma-3-12b completions before
launch.

## 9. Validator evaluation

A submission is evaluated **on receipt** — there is no batched pass. When a miner's endpoint returns a
submission, the validator scores it immediately against the prompt set active at that time:

1. Validate the submission (§7); reject (typed error) on any violation.
2. Register a forward hook that adds `alpha × direction` to the layer-32 residual stream at **every
   token position** during generation.
3. Generate completions for the day's prompts via **greedy decode** (`do_sample = false`,
   `num_beams = 1`) with a **fixed seed** (`1234`), `max_new_tokens = 64`, batch size 16, left padding,
   eager attention.
4. Score the completions with the active concept's pinned detector and scoring mode (§8).

The result is a `score ∈ [0, 1]`, returned with diagnostics that self-label the run (`device`,
`quantized`, `scoring_mode`, `detector_version`, `model_revision`, per-completion intensities). It is
applied directly to the incentive mechanism (§10).

## 10. Incentive and rewards

Each weekly concept is a winner-take-all competition following the standard Apex mechanism. A
submission is scored on receipt (§9); if it beats the current leader, it becomes the new leader and
holds the full emission for that concept until a higher-scoring submission replaces it. Apex applies
built-in decay to a standing leader. Winning submissions remain hidden until the weekly round
completes, then are published.

**This logic is owned by the parent Bittensor validator, not by this module** — the scorer only
returns the per-submission score that the incentive mechanism consumes ([§11](#11-scope-boundary)).

## 11. Scope boundary

The `concept-scorer` module is a **pure, stateless scorer** implementing §2–§9 and §12. The following
parts of the competition are deliberately owned by the **parent Bittensor validator**:

- **Weekly concept schedule (§4)** — which concept is active on a given date (one per week, no
  carryover). The scorer is concept-agnostic and rejects an off-schedule submission only via the
  `concept`-match check; it does not derive the active concept from the date.
- **Incentive / winner-take-all / decay / publication (§10)** — leaderboard, leader tracking,
  emission decay, hiding winners until a round closes.
- **Apex query protocol & rate-limits (§7)** — pulling submissions from miner endpoints and enforcing
  the one-submission-per-day-per-concept / four-per-day-per-hotkey caps.

## 12. Pre-launch requirements

- **Reference reproducibility smoke test** — the validator reproduces a known-good steering result
  (the weather-concept reference vector) end-to-end via `concept-scorer smoke`.
- **Held-out prompt pool** — `unsloth/alpaca-cleaned` pool assembled, frozen, and sha256-checked
  before Day 1.
- **Pipeline verified end-to-end** — submission validation, scoring (both modes), and the API are
  covered by the no-GPU test suite; the weather-reference smoke covers the GPU path. The GPU smoke
  is **backend-aware**: it enforces the calibrated absolute hit-rate floor only on the canonical
  CUDA + NF4 backend, and on the non-canonical MPS/CPU dev backend asserts only that steering lifts
  the weather rate over the `alpha=0` baseline (off-CUDA absolute scores are not canonical — §2).
- **Detection methods finalized and version-pinned** — all four concepts pinned (cake/hedging `v3`,
  medical `v2`, positive-sentiment `v3`), enforced against config at load.
- **Reproducibility configuration pinned** — model revision SHA, NF4/bf16, GPU architecture, and
  `transformers`/`bitsandbytes` versions, with build/load fail-fast guards on the unpinned-revision
  placeholder. The pinned `transformers` version must be confirmed on real CUDA hardware (the local
  dev pin is MPS-verified only).
- **Calibration on the canonical backend** — `alpha`, the per-concept `threshold`/`saturation`, and the
  weight tables tuned against real steered gemma-3-12b completions on CUDA + NF4, then frozen.
- **GPU-smoke floor recalibrated on CUDA NF4 (2026-06-02).** The old `CANONICAL_FLOOR = 0.25` was an
  MPS-era estimate, never CUDA-tested. An alpha sweep on CUDA NF4 shows the weather reference
  plateaus at **~0.18–0.20** (alpha 12k–16k) and **degenerates into repetition above ~20k**, so 0.25
  was unreachable. At alpha 12000: 0.180 (torch 2.5.1 / bnb 0.49.2), 0.193 (torch 2.11 / bnb 0.49.2),
  0.187 (vLLM). Kept `alpha=12000` (peak, safely pre-degeneration); lowered `CANONICAL_FLOOR` and the
  CLI smoke default to **0.15**. Also fixed the pin set that blocked the canonical CUDA stack:
  `transformers==5.9.0` needs `huggingface_hub>=1.5.0` (was `0.27.1`) and `bitsandbytes>=0.46.1`
  (was `0.45.0`) — now `huggingface_hub==1.17.0`, `bitsandbytes==0.49.2`. Re-confirm if the stack changes.

---

## Changes from the original draft

| Area | Original draft | As-built |
|------|----------------|----------|
| Detection | Birthday/medical/hedging regex libraries; positive-sentiment a pinned **classifier** | All four are **weighted lexicons**; positive-sentiment is **AFINN-111** (vendored, sha256-pinned) with phrase + negation handling |
| Scoring | `score = hit_rate` (uniform) | Per-concept `hit_rate` **or** `graded` (mean normalized intensity in `[0,1]`); `graded` is a deliberate, config-gated deviation |
| Schedule (§4) | Module mandates a 4-week rotation | Module is **concept-agnostic**; the schedule is owned by the validator |
| Incentive (§10) / query (§7) | Described as part of the competition | Explicitly owned by the **parent validator**; out of scope for this module |
| Backend | CUDA + NF4 only | CUDA + NF4 canonical; **MPS/CPU dev path** (unquantized bf16, non-reproducible, self-labeled) added |
| Revision/version pinning | Required | **Enforced** via build/load fail-fast guards; libraries pinned (Mac overlay marked dev-only) |
| Submission validation | Shape/dtype/norm/layer/concept | Same, plus non-finite, alpha-bounds, and unknown-concept rejection (typed errors) |
| Outputs | Hit rate | Per-completion intensity + `scoring_mode` + backend self-labeling (`device`/`quantized`); wire schema v2 |
