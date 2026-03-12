"""
Core VGGT inference pipeline — hardened, no network calls, no Gradio.
"""

import gc
import logging
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import AppConfig
from .security import (
    InputValidationError,
    validate_image_batch,
    load_npz_secure,
)

logger = logging.getLogger(__name__)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(cfg: AppConfig) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = mapping.get(cfg.gpu.dtype, torch.bfloat16)
    if dtype == torch.bfloat16 and torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] < 8:
            logger.warning("GPU does not support bfloat16, falling back to float16")
            dtype = torch.float16
    return dtype


def run_inference(
    model,
    image_dir: str,
    cfg: AppConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Run VGGT inference on a directory of images.

    Args:
        model: Loaded VGGT model
        image_dir: Directory containing image files
        cfg: App configuration
        output_dir: Where to save predictions.npz (optional)

    Returns:
        Dict of prediction arrays (numpy)
    """
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    device = get_device()
    dtype = get_dtype(cfg)

    # Discover and validate images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    # Filter to only allowed extensions
    allowed = set(cfg.limits.allowed_extensions)
    image_paths = [
        p for p in image_paths
        if os.path.splitext(p)[1].lower() in allowed
    ]

    if not image_paths:
        raise InputValidationError(f"No valid images found in {image_dir}")

    validate_image_batch(image_paths, cfg)
    logger.info("Processing %d images", len(image_paths))

    # Load and preprocess
    images = load_and_preprocess_images(image_paths).to(device)

    # Inference
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        raise InputValidationError(
            f"GPU out of memory with {len(image_paths)} images. "
            f"Reduce image count or resolution."
        )

    # Post-process: camera matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert to numpy
    results = {}
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            results[key] = val.cpu().numpy().squeeze(0)

    # Depth unprojection (more accurate than point map)
    depth_map = results.get("depth")
    if depth_map is not None:
        world_pts = unproject_depth_map_to_point_map(
            depth_map, results["extrinsic"], results["intrinsic"]
        )
        results["world_points_from_depth"] = world_pts

    # Cleanup GPU
    del predictions, images
    torch.cuda.empty_cache()
    gc.collect()

    # Save predictions
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        npz_path = os.path.join(output_dir, "predictions.npz")
        np.savez(npz_path, **results)
        logger.info("Saved predictions to %s", npz_path)

    return results


def extract_point_cloud(
    predictions: Dict[str, np.ndarray],
    use_depth: bool = True,
    confidence_pct: float = 50.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract filtered point cloud from VGGT predictions.

    Returns (points_xyz [N,3], colors_rgb [N,3] or None)
    """
    # Pick point source
    if use_depth and "world_points_from_depth" in predictions:
        pts = predictions["world_points_from_depth"]
    elif "world_points" in predictions:
        pts = predictions["world_points"]
    else:
        raise ValueError("No point cloud in predictions")

    if pts.ndim == 4:
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 3:
        pts = pts.reshape(-1, 3)

    # Colors
    colors = None
    if "images" in predictions:
        imgs = predictions["images"]
        if imgs.ndim == 4:
            if imgs.shape[-1] == 3:
                colors = imgs.reshape(-1, 3)
            elif imgs.shape[1] == 3:
                colors = np.transpose(imgs, (0, 2, 3, 1)).reshape(-1, 3)
        if colors is not None and colors.max() > 1.0:
            colors = colors / 255.0

    # Confidence filter
    for conf_key in ("depth_conf", "world_points_conf"):
        if conf_key in predictions:
            conf = predictions[conf_key].reshape(-1)
            if len(conf) >= len(pts):
                conf = conf[:len(pts)]
                threshold = np.percentile(conf, confidence_pct)
                mask = conf >= threshold
                pts = pts[mask]
                if colors is not None and len(colors) >= len(mask):
                    colors = colors[:len(mask)][mask]
                break

    # Remove invalid
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if colors is not None:
        colors = colors[:len(valid)][valid]

    # Statistical outlier removal
    if len(pts) > 100:
        centroid = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        cutoff = np.percentile(dists, 99)
        mask = dists < cutoff
        pts = pts[mask]
        if colors is not None:
            colors = colors[mask]

    logger.info("Point cloud: %d points", len(pts))
    return pts, colors
