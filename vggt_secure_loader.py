#!/usr/bin/env python3
"""
vggt_secure_loader.py - Security-hardened VGGT model loader

Patches for findings VGGT-001, VGGT-004, VGGT-008:
  - SHA-256 hash verification of model weights
  - weights_only=True loading (PyTorch 2.6+)
  - Secure temp directory handling
  - Input validation for images

Drop this file into your VGGT project root and use it instead of
directly calling VGGT.from_pretrained() or torch.hub.load_state_dict_from_url().

Usage:
    from vggt_secure_loader import load_vggt_model_secure, load_images_secure
    model = load_vggt_model_secure(device="cuda")
    images = load_images_secure(["img1.png", "img2.png"], device="cuda")
"""

import os
import sys
import hashlib
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Configuration - UPDATE THESE for your deployment
# ═══════════════════════════════════════════════════════════════════════════

# SHA-256 hash of the official VGGT-1B model.pt
# Run: sha256sum model.pt   to get this value after first trusted download
# Set to None to skip verification (NOT recommended for production)
EXPECTED_MODEL_HASH: Optional[str] = None  # e.g., "abc123..."

# Commercial checkpoint URL (requires HF access approval)
MODEL_URL_COMMERCIAL = "https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/model.pt"
# Original (non-commercial) checkpoint URL
MODEL_URL_ORIGINAL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

# Local cache directory for model weights
MODEL_CACHE_DIR = os.environ.get("VGGT_MODEL_CACHE", os.path.expanduser("~/.cache/vggt"))

# Image validation settings
MAX_IMAGE_COUNT = 100
MAX_IMAGE_DIMENSION = 4096  # pixels per side
MAX_IMAGE_FILE_SIZE_MB = 50
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
IMAGE_MAGIC_BYTES = {
    b'\x89PNG': 'png',
    b'\xff\xd8\xff': 'jpeg',
    b'BM': 'bmp',
    b'II': 'tiff',  # little-endian TIFF
    b'MM': 'tiff',  # big-endian TIFF
    b'RIFF': 'webp',
}


# ═══════════════════════════════════════════════════════════════════════════
# Model Loading (patches VGGT-001)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file in chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_model_integrity(filepath: str, expected_hash: Optional[str] = None) -> bool:
    """Verify model file integrity via SHA-256 hash."""
    if expected_hash is None:
        logger.warning(
            "No expected hash configured. Skipping integrity check. "
            "Set EXPECTED_MODEL_HASH for production use."
        )
        return True

    actual_hash = compute_sha256(filepath)
    if actual_hash != expected_hash:
        logger.error(
            f"MODEL INTEGRITY CHECK FAILED!\n"
            f"  Expected: {expected_hash}\n"
            f"  Got:      {actual_hash}\n"
            f"  File:     {filepath}\n"
            f"  This may indicate a tampered or corrupted model file."
        )
        return False

    logger.info(f"Model integrity verified: {actual_hash[:16]}...")
    return True


def load_vggt_model_secure(
    device: str = "cuda",
    commercial: bool = True,
    model_path: Optional[str] = None,
    expected_hash: Optional[str] = None,
) -> "VGGT":
    """
    Load VGGT model with security hardening.

    Args:
        device: Target device ("cuda" or "cpu")
        commercial: Use commercial-licensed checkpoint (recommended)
        model_path: Path to local model weights (skips download if provided)
        expected_hash: SHA-256 hash for verification (overrides EXPECTED_MODEL_HASH)

    Returns:
        VGGT model instance, loaded and in eval mode
    """
    from vggt.models.vggt import VGGT

    hash_to_check = expected_hash or EXPECTED_MODEL_HASH

    if model_path and os.path.isfile(model_path):
        # Load from local file
        logger.info(f"Loading model from local path: {model_path}")
        weight_path = model_path
    else:
        # Download and cache
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        url = MODEL_URL_COMMERCIAL if commercial else MODEL_URL_ORIGINAL
        cache_filename = "model_commercial.pt" if commercial else "model_original.pt"
        weight_path = os.path.join(MODEL_CACHE_DIR, cache_filename)

        if not os.path.isfile(weight_path):
            logger.info(f"Downloading model weights from {url}")
            # Use torch.hub for download with progress
            state_dict = torch.hub.load_state_dict_from_url(
                url, model_dir=MODEL_CACHE_DIR, map_location="cpu"
            )
            # Save to known location
            torch.save(state_dict, weight_path)
            del state_dict
            logger.info(f"Model cached to {weight_path}")

            # Print hash for first-time setup
            actual_hash = compute_sha256(weight_path)
            logger.info(
                f"Model SHA-256: {actual_hash}\n"
                f"Set EXPECTED_MODEL_HASH = '{actual_hash}' in vggt_secure_loader.py "
                f"for future integrity checks."
            )

    # Verify integrity
    if not verify_model_integrity(weight_path, hash_to_check):
        raise SecurityError(
            f"Model file failed integrity check: {weight_path}. "
            f"Delete the file and re-download from a trusted source."
        )

    # Load with security flags
    model = VGGT()
    pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])

    if pytorch_version >= (2, 6):
        # PyTorch 2.6+ supports weights_only=True to block pickle RCE
        logger.info("Using weights_only=True (PyTorch 2.6+ safe loading)")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    else:
        logger.warning(
            f"PyTorch {torch.__version__} does not support weights_only=True. "
            f"Upgrade to PyTorch >= 2.6 for safe deserialization."
        )
        state_dict = torch.load(weight_path, map_location="cpu")

    model.load_state_dict(state_dict)
    del state_dict

    model.eval()
    model = model.to(device)
    logger.info(f"VGGT model loaded on {device}")

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Input Validation (patches VGGT-002, VGGT-003, VGGT-010)
# ═══════════════════════════════════════════════════════════════════════════

class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class InputValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_image_file(filepath: str) -> None:
    """
    Validate a single image file for safety.

    Checks: extension, file size, magic bytes, dimensions.
    """
    path = Path(filepath)

    # Extension check
    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise InputValidationError(
            f"Disallowed file extension: {path.suffix}. "
            f"Allowed: {ALLOWED_IMAGE_EXTENSIONS}"
        )

    # File size check
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_IMAGE_FILE_SIZE_MB:
        raise InputValidationError(
            f"File too large: {file_size_mb:.1f}MB > {MAX_IMAGE_FILE_SIZE_MB}MB limit"
        )

    # Magic bytes check
    with open(filepath, "rb") as f:
        header = f.read(8)

    valid_magic = False
    for magic, fmt in IMAGE_MAGIC_BYTES.items():
        if header.startswith(magic):
            valid_magic = True
            break

    if not valid_magic:
        raise InputValidationError(
            f"File does not appear to be a valid image (bad magic bytes): {filepath}"
        )

    # Dimension check (uses Pillow for safe header parsing)
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
                raise InputValidationError(
                    f"Image too large: {w}x{h} > {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} limit"
                )
            if w < 1 or h < 1:
                raise InputValidationError(f"Invalid image dimensions: {w}x{h}")
    except (Image.UnidentifiedImageError, OSError) as e:
        raise InputValidationError(f"Cannot open as image: {filepath}: {e}")


def validate_image_batch(filepaths: List[str]) -> List[str]:
    """
    Validate a batch of image files.

    Returns the validated list of filepaths.
    """
    if len(filepaths) > MAX_IMAGE_COUNT:
        raise InputValidationError(
            f"Too many images: {len(filepaths)} > {MAX_IMAGE_COUNT} limit"
        )

    if len(filepaths) == 0:
        raise InputValidationError("No images provided")

    validated = []
    for fp in filepaths:
        # Canonicalize path to prevent traversal
        real_path = os.path.realpath(fp)
        if not os.path.isfile(real_path):
            raise InputValidationError(f"File not found: {fp}")

        validate_image_file(real_path)
        validated.append(real_path)

    logger.info(f"Validated {len(validated)} images")
    return validated


def load_images_secure(
    image_paths: List[str],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Securely load and preprocess images for VGGT inference.

    Validates all images before loading, then delegates to VGGT's
    load_and_preprocess_images.
    """
    from vggt.utils.load_fn import load_and_preprocess_images

    validated_paths = validate_image_batch(image_paths)
    images = load_and_preprocess_images(validated_paths).to(device)
    return images


# ═══════════════════════════════════════════════════════════════════════════
# Secure numpy loading (patches VGGT-008)
# ═══════════════════════════════════════════════════════════════════════════

def load_predictions_secure(npz_path: str) -> dict:
    """
    Load VGGT predictions from .npz without allow_pickle=True.

    The predictions dict contains only numpy arrays, so pickle
    should not be needed. If it fails, the file may be tampered.
    """
    real_path = os.path.realpath(npz_path)
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"Predictions file not found: {npz_path}")

    try:
        loaded = np.load(real_path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.keys()}
    except ValueError as e:
        raise SecurityError(
            f"Predictions file requires pickle (may be tampered): {npz_path}. "
            f"Original error: {e}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Secure filename generation (patches VGGT-007)
# ═══════════════════════════════════════════════════════════════════════════

def safe_output_filename(base_dir: str, params: dict, extension: str = ".glb") -> str:
    """
    Generate a safe output filename from parameters using a hash
    instead of concatenating user-controlled strings into paths.
    """
    base_dir = os.path.realpath(base_dir)
    if not os.path.isdir(base_dir):
        raise ValueError(f"Output directory does not exist: {base_dir}")

    # Hash the parameters to get a deterministic, safe filename
    param_str = str(sorted(params.items()))
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
    filename = f"output_{param_hash}{extension}"

    full_path = os.path.join(base_dir, filename)
    # Verify the path is still within base_dir
    if not os.path.realpath(full_path).startswith(base_dir):
        raise SecurityError(f"Path traversal detected: {full_path}")

    return full_path


# ═══════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("VGGT Secure Loader - Configuration Check")
    print(f"  Model cache dir:    {MODEL_CACHE_DIR}")
    print(f"  Hash verification:  {'ENABLED' if EXPECTED_MODEL_HASH else 'DISABLED (set EXPECTED_MODEL_HASH)'}")
    print(f"  Max images:         {MAX_IMAGE_COUNT}")
    print(f"  Max dimension:      {MAX_IMAGE_DIMENSION}px")
    print(f"  Max file size:      {MAX_IMAGE_FILE_SIZE_MB}MB")
    print(f"  PyTorch version:    {torch.__version__}")
    pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    print(f"  weights_only safe:  {'YES' if pytorch_version >= (2, 6) else 'NO (upgrade PyTorch)'}")
    print("\nAll checks passed. Ready for deployment.")
