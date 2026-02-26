"""FastAPI server for PINN inference. Run with: uvicorn main:app --reload"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from equations import EQUATIONS, get_equation, list_equations
from model import PhysicsInformedNN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
SPATIAL_POINTS_DEFAULT = 100
SPATIAL_POINTS_MAX = 500
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:8000",
).split(",")

loaded_models: dict[str, PhysicsInformedNN] = {}
model_loaded_at: float | None = None


def _model_path(eq_key: str) -> Path:
    """Return the expected model file path for an equation key."""
    return Path(MODEL_DIR) / f"{eq_key}_model.pth"


def _legacy_model_path() -> Path:
    """Old single-model path for backward compatibility."""
    return Path("burgers_model.pth")


def load_models() -> None:
    """Scan for trained models and load them."""
    global model_loaded_at
    loaded_models.clear()
    count = 0

    for eq_key, eq in EQUATIONS.items():
        path = _model_path(eq_key)
        # Backward compat: if burgers model is at old root location
        if eq_key == "burgers" and not path.exists() and _legacy_model_path().exists():
            path = _legacy_model_path()

        if path.exists():
            try:
                m = PhysicsInformedNN(
                    hidden_size=eq.recommended_hidden,
                    num_layers=eq.recommended_layers,
                )
                m.load_state_dict(torch.load(str(path), weights_only=True))
                m.eval()
                loaded_models[eq_key] = m
                count += 1
                logger.info("Loaded model for '%s' from %s", eq.name, path)
            except Exception:
                logger.exception("Failed to load model for '%s'", eq_key)

    if count == 0:
        logger.warning("No trained models found. Run: python pinn_model.py --equation all")
    else:
        logger.info("Loaded %d model(s)", count)
    model_loaded_at = time.time()


def startup_event() -> None:
    """Called on startup and in tests."""
    load_models()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    startup_event()
    yield


app = FastAPI(
    title="PINN PDE Solver",
    description="Multi-equation Physics-Informed Neural Network solver.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SnapshotRequest(BaseModel):
    t: float = Field(..., ge=0.0, le=1.0, description="Time value in [0, 1]")
    n_points: int = Field(
        default=SPATIAL_POINTS_DEFAULT, ge=10, le=SPATIAL_POINTS_MAX,
        description=f"Number of spatial points (10–{SPATIAL_POINTS_MAX})",
    )


class SnapshotResponse(BaseModel):
    x: list[float]
    u: list[float]
    t: float
    n_points: int
    equation: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    models_available: int
    uptime_seconds: float | None


@lru_cache(maxsize=512)
def _predict_cached(
    eq_key: str, t_rounded: float, n_points: int
) -> tuple[list[float], list[float]]:
    model = loaded_models[eq_key]
    eq = EQUATIONS[eq_key]
    x_np = np.linspace(eq.x_min, eq.x_max, n_points).astype(np.float32)
    x_tensor = torch.from_numpy(x_np).view(-1, 1)
    t_tensor = torch.full_like(x_tensor, t_rounded)

    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor)

    return x_np.tolist(), u_pred.flatten().tolist()


@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    uptime = round(time.time() - model_loaded_at, 1) if model_loaded_at else None
    return HealthResponse(
        status="healthy" if loaded_models else "degraded",
        models_loaded=sorted(loaded_models.keys()),
        models_available=len(loaded_models),
        uptime_seconds=uptime,
    )


@app.get("/api/v1/equations")
def get_equations():
    """Return metadata for all registered equations, with trained status."""
    eq_list = list_equations()
    for eq_info in eq_list:
        eq_info["trained"] = eq_info["key"] in loaded_models
    return eq_list


@app.post("/api/v1/predict/{equation_key}", response_model=SnapshotResponse)
def predict_snapshot(equation_key: str, data: SnapshotRequest):
    """Predict u(x) at time t for a given equation."""
    if equation_key not in EQUATIONS:
        available = ", ".join(sorted(EQUATIONS.keys()))
        raise HTTPException(404, f"Unknown equation '{equation_key}'. Available: {available}")

    if equation_key not in loaded_models:
        raise HTTPException(
            503,
            f"Model for '{equation_key}' not trained. "
            f"Run: python pinn_model.py --equation {equation_key}",
        )

    t_rounded = round(data.t, 2)
    x_list, u_list = _predict_cached(equation_key, t_rounded, data.n_points)
    return SnapshotResponse(
        x=x_list, u=u_list, t=t_rounded,
        n_points=data.n_points, equation=equation_key,
    )


# Legacy endpoint (backward compat)
@app.post("/predict_snapshot")
@app.post("/api/v1/predict_snapshot")
def predict_snapshot_legacy(data: SnapshotRequest):
    return predict_snapshot("burgers", data)


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
