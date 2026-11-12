from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import get_settings
from app.state import get_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Initialize state and provider on startup
    state = get_state()
    settings = get_settings()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="LLM Smart Router v2",
        description="FastAPI LLM proxy that dynamically routes requests based on prompt classification.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Configure CORS for local development and integration with frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Routed-Category", "X-Routed-Model"],
    )

    app.include_router(api_router)

    return app


app = create_app()
