from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jose import JWTError

from .api.v1.router import api_router
from .config import settings
from .core.exceptions import jwt_exception_handler

UPLOADS_DIR = Path("data/uploads")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from .tools.rags.risk_rag import get_risk_rag
        from .tools.rags.incidents_rag import get_incidents_rag
        from .tools.rags.framework_rag import get_framework_rag
        get_risk_rag()
        get_incidents_rag()
        get_framework_rag()
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("RAG initialization failed (non-fatal): %s", exc)

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Ethics Auditor API",
        description="Multi-agent AI ethics auditing system",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(JWTError, jwt_exception_handler)  # type: ignore[arg-type]
    app.include_router(api_router, prefix="/api/v1")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

    return app


app = create_app()
