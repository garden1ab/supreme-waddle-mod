"""
Secure REST API server replacing Gradio.

- API key authentication on all endpoints
- Rate limiting per key
- Input validation before any processing
- No stack traces to clients
- Security headers on all responses
- Localhost binding by default
- Automatic temp cleanup
"""

import asyncio
import gc
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import AppConfig, load_config
from .security import (
    SecurityError,
    InputValidationError,
    RateLimitError,
    TempDirManager,
    TokenBucketRateLimiter,
    validate_image_bytes,
    validate_image_file,
    validate_image_batch,
    load_model_secure,
    load_npz_secure,
    sanitize_filename,
)
from .inference import run_inference, extract_point_cloud
from .solidworks import export_format, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

# ── Global state ────────────────────────────────────────────────────────
_cfg: Optional[AppConfig] = None
_model = None
_temp_mgr: Optional[TempDirManager] = None
_rate_limiter: Optional[TokenBucketRateLimiter] = None
_cleanup_task = None


# ── Lifecycle ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _model, _temp_mgr, _rate_limiter, _cleanup_task

    _cfg = load_config()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, _cfg.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load model
    logger.info("Loading VGGT model...")
    _model = load_model_secure(_cfg)
    logger.info("Model ready.")

    # Temp manager
    _temp_mgr = TempDirManager(_cfg.temp.base_dir, _cfg.temp.ttl_seconds)

    # Rate limiter
    _rate_limiter = TokenBucketRateLimiter(_cfg.server.rate_limit_rpm)

    # Background cleanup
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)  # every 5 min
            _temp_mgr.cleanup_expired()

    _cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    # Shutdown
    if _cleanup_task:
        _cleanup_task.cancel()
    logger.info("Shutting down.")


app = FastAPI(
    title="VGGT-Secure API",
    version="1.0.0",
    docs_url=None,     # Disable Swagger UI in production
    redoc_url=None,     # Disable ReDoc in production
    openapi_url=None,   # Disable OpenAPI schema in production
    lifespan=lifespan,
)


# ── Security Middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store"
    # Remove server header
    response.headers.pop("server", None)
    return response


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Never leak stack traces to clients."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )
    if isinstance(exc, InputValidationError):
        return JSONResponse(status_code=400, content={"error": str(exc)})
    if isinstance(exc, RateLimitError):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    # Log internally, return generic message
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal processing error"},
    )


# ── Auth Dependency ─────────────────────────────────────────────────────

async def verify_api_key(request: Request):
    """Verify API key from Authorization header."""
    if not _cfg or not _cfg.api_key:
        return  # Auth disabled (local CLI mode)

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = auth[7:]
    if token != _cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Rate limit
    if _rate_limiter and not _rate_limiter.check(token):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ── Response Models ─────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool


class ReconstructResponse(BaseModel):
    job_id: str
    num_images: int
    num_points: int
    message: str


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health():
    """Health check — no auth required."""
    return HealthResponse(
        status="ok",
        gpu_available=torch.cuda.is_available(),
        model_loaded=_model is not None,
    )


@app.post("/api/v1/reconstruct", dependencies=[Depends(verify_api_key)])
async def reconstruct(images: List[UploadFile] = File(...)):
    """
    Upload images, run 3D reconstruction, return predictions.

    Returns a job_id that can be used with /export/{format} endpoint.
    """
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create temp dir
    work_dir = _temp_mgr.create()
    image_dir = os.path.join(work_dir, "images")
    job_id = os.path.basename(work_dir)

    try:
        # Receive and validate uploads
        if len(images) > _cfg.limits.max_images:
            raise InputValidationError(
                f"Too many images: {len(images)} > {_cfg.limits.max_images}"
            )

        saved_paths = []
        total_bytes = 0
        for upload in images:
            data = await upload.read()
            total_bytes += len(data)

            if total_bytes > _cfg.limits.max_upload_total_mb * 1024 * 1024:
                raise InputValidationError("Total upload size exceeded")

            safe_name = sanitize_filename(upload.filename or "image.jpg")
            validate_image_bytes(data, safe_name, _cfg)

            dest = os.path.join(image_dir, f"{uuid.uuid4().hex[:12]}_{safe_name}")
            with open(dest, "wb") as f:
                f.write(data)

            validate_image_file(dest, _cfg)
            saved_paths.append(dest)

        # Run inference
        predictions = run_inference(_model, image_dir, _cfg, output_dir=work_dir)

        pts, _ = extract_point_cloud(predictions)

        return ReconstructResponse(
            job_id=job_id,
            num_images=len(saved_paths),
            num_points=len(pts),
            message="Reconstruction complete. Use /export/{format} with this job_id.",
        )

    except InputValidationError:
        _temp_mgr.remove(work_dir)
        raise
    except Exception:
        _temp_mgr.remove(work_dir)
        raise


@app.post("/api/v1/export/{fmt}", dependencies=[Depends(verify_api_key)])
async def export_cad(fmt: str, job_id: str):
    """
    Export a previous reconstruction to a SolidWorks-compatible format.

    Formats: stl, obj, ply, step, iges
    """
    fmt = fmt.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format. Use: {SUPPORTED_FORMATS}")

    # Locate job
    work_dir = os.path.join(_temp_mgr.base_dir, job_id)
    work_dir = os.path.realpath(work_dir)
    if not work_dir.startswith(os.path.realpath(_temp_mgr.base_dir)):
        raise HTTPException(400, "Invalid job_id")

    npz_path = os.path.join(work_dir, "predictions.npz")
    if not os.path.isfile(npz_path):
        raise HTTPException(404, "Job not found or expired")

    predictions = load_npz_secure(npz_path)
    pts, colors = extract_point_cloud(predictions)

    export_dir = os.path.join(work_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)

    output_path = export_format(fmt, pts, colors, export_dir)

    ext_map = {
        "stl": "application/octet-stream",
        "obj": "text/plain",
        "ply": "application/octet-stream",
        "step": "application/step",
        "iges": "application/iges",
    }

    return FileResponse(
        output_path,
        media_type=ext_map.get(fmt, "application/octet-stream"),
        filename=f"vggt_export.{fmt}",
    )


# ── Entry point ─────────────────────────────────────────────────────────

def run_server():
    """Start the API server."""
    import uvicorn

    cfg = load_config()
    logger.info(
        "Starting VGGT-Secure API on %s:%d",
        cfg.server.host, cfg.server.port
    )

    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
        timeout_keep_alive=30,
        access_log=False,       # We handle logging ourselves
        server_header=False,    # Don't leak server info
    )
