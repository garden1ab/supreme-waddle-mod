"""
Security module: model integrity, input validation, path safety, rate limiting.
Patches: VGGT-001 through VGGT-012.
"""

import hashlib
import logging
import os
import re
import shutil
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .config import AppConfig

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════

class SecurityError(Exception):
    """A security check has failed. Do not expose details to client."""
    pass


class InputValidationError(ValueError):
    """Input did not pass validation. Safe to expose message to client."""
    pass


class RateLimitError(Exception):
    """Too many requests."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Model Integrity (VGGT-001)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 of a file in 8KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_file(filepath: str, expected_hash: str) -> bool:
    """Verify model weights against expected SHA-256."""
    if not expected_hash:
        logger.warning("No model hash configured — integrity check SKIPPED. Set VGGT_MODEL_HASH.")
        return True
    actual = compute_sha256(filepath)
    if actual != expected_hash:
        logger.error(
            "MODEL INTEGRITY FAILED: hash mismatch. "
            "Expected %s, got %s. File may be tampered.",
            expected_hash[:16], actual[:16]
        )
        return False
    logger.info("Model integrity verified: %s", actual[:16])
    return True


def load_model_secure(cfg: AppConfig) -> "VGGT":
    """
    Load VGGT model with all security hardening:
    - Offline-first (no runtime downloads)
    - SHA-256 verification
    - weights_only=True on PyTorch >= 2.6
    """
    from vggt.models.vggt import VGGT

    weight_path = cfg.model.path
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Model weights not found at {weight_path}. "
            f"Run 'python -m vggt_secure.cli download-model' first."
        )

    if not verify_model_file(weight_path, cfg.model.expected_hash):
        raise SecurityError("Model file failed integrity check. Aborting.")

    model = VGGT()
    pt_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])

    if pt_version >= (2, 6):
        logger.info("Loading with weights_only=True (PyTorch %s)", torch.__version__)
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    else:
        logger.warning(
            "PyTorch %s < 2.6: weights_only not supported. Upgrade recommended.",
            torch.__version__,
        )
        state_dict = torch.load(weight_path, map_location="cpu")

    model.load_state_dict(state_dict)
    del state_dict

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info("Model loaded on %s", device)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Input Validation (VGGT-002, VGGT-003, VGGT-010)
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_MAGIC = {
    b"\x89PNG": "png",
    b"\xff\xd8\xff": "jpeg",
    b"BM": "bmp",
    b"II": "tiff",
    b"MM": "tiff",
    b"RIFF": "webp",
}


def validate_image_bytes(data: bytes, filename: str, cfg: AppConfig) -> None:
    """Validate raw image bytes before writing to disk."""
    # Size
    size_mb = len(data) / (1024 * 1024)
    if size_mb > cfg.limits.max_file_size_mb:
        raise InputValidationError(
            f"File too large: {size_mb:.1f}MB exceeds {cfg.limits.max_file_size_mb}MB limit"
        )

    # Extension
    ext = Path(filename).suffix.lower()
    if ext not in cfg.limits.allowed_extensions:
        raise InputValidationError(f"File type '{ext}' not allowed")

    # Magic bytes
    valid = any(data[:len(magic)].startswith(magic) for magic in IMAGE_MAGIC)
    if not valid:
        raise InputValidationError("File content does not match any known image format")


def validate_image_file(filepath: str, cfg: AppConfig) -> None:
    """Validate an image file on disk: dimensions and parseability."""
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            if w > cfg.limits.max_resolution or h > cfg.limits.max_resolution:
                raise InputValidationError(
                    f"Image {w}x{h} exceeds {cfg.limits.max_resolution}px limit"
                )
            if w < 1 or h < 1:
                raise InputValidationError(f"Invalid dimensions: {w}x{h}")
            # Force full decode to catch truncated/corrupt images
            img.load()
    except (Image.UnidentifiedImageError, OSError) as e:
        raise InputValidationError(f"Cannot parse image: {e}")


def validate_image_batch(filepaths: List[str], cfg: AppConfig) -> None:
    """Validate a batch of image files."""
    if len(filepaths) == 0:
        raise InputValidationError("No images provided")
    if len(filepaths) > cfg.limits.max_images:
        raise InputValidationError(
            f"Too many images: {len(filepaths)} exceeds {cfg.limits.max_images} limit"
        )

    total_bytes = 0
    for fp in filepaths:
        real = os.path.realpath(fp)
        if not os.path.isfile(real):
            raise InputValidationError("Image file not found")
        total_bytes += os.path.getsize(real)
        validate_image_file(real, cfg)

    total_mb = total_bytes / (1024 * 1024)
    if total_mb > cfg.limits.max_upload_total_mb:
        raise InputValidationError(
            f"Total upload {total_mb:.0f}MB exceeds {cfg.limits.max_upload_total_mb}MB limit"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Secure Temp Directories (VGGT-006)
# ═══════════════════════════════════════════════════════════════════════════

class TempDirManager:
    """Manages temporary directories with UUID names and auto-cleanup."""

    def __init__(self, base_dir: str, ttl_seconds: int = 3600):
        self.base_dir = os.path.realpath(base_dir)
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        os.makedirs(self.base_dir, mode=0o700, exist_ok=True)

    def create(self) -> str:
        """Create a new temp directory with a UUID name."""
        dir_name = str(uuid.uuid4())
        path = os.path.join(self.base_dir, dir_name)
        os.makedirs(path, mode=0o700)
        os.makedirs(os.path.join(path, "images"), mode=0o700)
        return path

    def cleanup_expired(self) -> int:
        """Remove directories older than TTL. Returns count removed."""
        removed = 0
        now = time.time()
        with self._lock:
            try:
                for entry in os.scandir(self.base_dir):
                    if entry.is_dir():
                        age = now - entry.stat().st_mtime
                        if age > self.ttl:
                            shutil.rmtree(entry.path, ignore_errors=True)
                            removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Cleaned up %d expired temp dirs", removed)
        return removed

    def remove(self, path: str) -> None:
        """Immediately remove a temp directory."""
        real = os.path.realpath(path)
        if not real.startswith(self.base_dir):
            raise SecurityError("Path traversal blocked in temp cleanup")
        shutil.rmtree(real, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# Safe Filename Generation (VGGT-007)
# ═══════════════════════════════════════════════════════════════════════════

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


def sanitize_filename(name: str) -> str:
    """Strip dangerous characters from a filename."""
    stem = Path(name).stem
    ext = Path(name).suffix.lower()
    clean = _SAFE_FILENAME_RE.sub("_", stem)[:100]
    return f"{clean}{ext}" if clean else f"file{ext}"


def safe_output_path(base_dir: str, params: dict, extension: str) -> str:
    """Generate a safe output filename from a hash of parameters."""
    base_dir = os.path.realpath(base_dir)
    param_hash = hashlib.sha256(
        str(sorted(params.items())).encode()
    ).hexdigest()[:16]
    filename = f"output_{param_hash}{extension}"
    full = os.path.join(base_dir, filename)
    if not os.path.realpath(full).startswith(base_dir):
        raise SecurityError("Path traversal in output path")
    return full


# ═══════════════════════════════════════════════════════════════════════════
# Secure Numpy Loading (VGGT-008)
# ═══════════════════════════════════════════════════════════════════════════

def load_npz_secure(path: str) -> Dict[str, np.ndarray]:
    """Load .npz without allow_pickle (blocks arbitrary code execution)."""
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        data = np.load(real, allow_pickle=False)
        return {k: data[k] for k in data.keys()}
    except ValueError:
        raise SecurityError(
            "NPZ file contains pickled objects — may be tampered. Refusing to load."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Rate Limiter (VGGT-003)
# ═══════════════════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """Per-key token bucket rate limiter."""

    def __init__(self, rpm: int):
        self.capacity = rpm
        self.refill_rate = rpm / 60.0  # tokens per second
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            elapsed = now - last
            tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                return True
            self._buckets[key] = (tokens, now)
            return False
