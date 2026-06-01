"""FastAPI app factory.

The app holds a single warm :class:`ModelRuntime` plus the frozen :class:`PromptPool`,
loaded once in the lifespan handler. A module-level state object exposes them to routes
and is overridable in unit tests (inject a fake runtime, skip model loading).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI

from ..config import Settings, get_settings
from ..prompts import PromptPool


@dataclass
class AppState:
    settings: Settings
    runtime: object | None = None
    pool: PromptPool | None = None
    lock: asyncio.Lock | None = None
    load_model: bool = True


def create_app(state: AppState | None = None) -> FastAPI:
    if state is None:
        state = AppState(settings=get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.lock = asyncio.Lock()
        if state.pool is None:
            state.pool = PromptPool.from_jsonl(state.settings.prompts.pool_path)
        if state.load_model and state.runtime is None:
            from ..backends import build_backend

            state.runtime = build_backend(state.settings)
        yield

    app = FastAPI(title="concept-scorer", lifespan=lifespan)
    app.state.scorer = state

    from .routes import router

    app.include_router(router)
    return app
