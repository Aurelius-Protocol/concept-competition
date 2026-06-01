"""Pluggable generation backends.

Both backends expose the same surface used by the scorer: ``load()``, ``ready``,
``model_revision``, and ``generate(instructions, submission) -> list[str]``.

* The in-process :class:`~concept_scorer.model_runtime.ModelRuntime` is the **only**
  backend that can apply the competition's layer-32 residual steering — it registers a
  PyTorch forward hook on the model's hidden states (white-box access).

* :class:`OpenAIBackend` talks to any OpenAI-compatible server (e.g. **LM Studio**) over
  HTTP. That API is **black-box** (text in / text out): there is no endpoint to inject a
  steering vector into the residual stream, so this backend **cannot apply steering** and
  therefore cannot produce a valid *steered* competition score. It refuses a nonzero-alpha
  request unless explicitly allowed to run an UNSTEERED baseline. Use it for plumbing
  checks and baselines only.
"""

from __future__ import annotations

from .config import Settings


class SteeringUnsupported(RuntimeError):
    """Raised when a steered (alpha != 0) score is requested from a backend that can't steer."""


class OpenAIBackend:
    """Black-box generation via an OpenAI-compatible endpoint (e.g. LM Studio). No steering."""

    def __init__(self, settings: Settings):
        self.settings = settings
        rt = settings.runtime
        if not rt.openai_base_url:
            raise ValueError(
                "the 'openai' backend needs CONCEPT_SCORER_OPENAI_BASE_URL "
                "(e.g. http://localhost:1234/v1 for LM Studio)"
            )
        self.base_url = rt.openai_base_url
        self.model = rt.openai_model
        self.api_key = rt.openai_api_key
        self.allow_unsteered = rt.allow_unsteered
        self._client = None
        self.ready = False

    def load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        if not self.model:
            # Default to whatever model the server currently has loaded.
            served = self._client.models.list()
            if not served.data:
                raise RuntimeError(f"no model loaded at {self.base_url}")
            self.model = served.data[0].id
        self.ready = True

    @property
    def model_revision(self) -> str:
        return f"openai:{self.model}"

    def generate(self, instructions: list[str], submission) -> list[str]:
        if abs(float(submission.alpha)) > 0.0 and not self.allow_unsteered:
            raise SteeringUnsupported(
                f"the external/LM Studio backend cannot apply residual steering "
                f"(submission alpha={submission.alpha}); its API is black-box text-in/text-out. "
                f"Use the in-process local backend for steered scoring, or pass --baseline to run "
                f"an UNSTEERED baseline (not a valid competition score)."
            )
        gen = self.settings.generation
        out: list[str] = []
        for instruction in instructions:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": instruction}],
                temperature=0,  # greedy
                max_tokens=gen.max_new_tokens,
                seed=gen.seed,
            )
            out.append((resp.choices[0].message.content or "").strip())
        return out


def build_backend(settings: Settings):
    """Construct and ``load()`` the backend selected by ``settings.runtime.backend``."""
    if settings.runtime.backend == "openai":
        backend = OpenAIBackend(settings)
    else:
        from .model_runtime import ModelRuntime

        backend = ModelRuntime(settings)
    backend.load()
    return backend
