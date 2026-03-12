"""
CLI tool for VGGT-Secure.

Usage:
    python -m vggt_secure.cli download-model
    python -m vggt_secure.cli reconstruct --scene_dir ./images/ --output ./results/
    python -m vggt_secure.cli export --input ./results/predictions.npz --format stl
    python -m vggt_secure.cli serve
"""

import argparse
import hashlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def cmd_download_model(args):
    """Download model weights and print SHA-256 for config."""
    import urllib.request
    from .config import load_config
    from .security import compute_sha256

    cfg = load_config()
    dest = cfg.model.path
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    url = cfg.model.commercial_url if cfg.model.commercial else cfg.model.original_url

    if os.path.isfile(dest):
        sha = compute_sha256(dest)
        print(f"Model already exists: {dest}")
        print(f"SHA-256: {sha}")
        print(f"\nSet environment variable: VGGT_MODEL_HASH={sha}")
        return

    print(f"Downloading model from {url}")
    print(f"Destination: {dest}")
    print("This may take a while (~5GB)...")

    urllib.request.urlretrieve(url, dest, _download_progress)
    print()

    sha = compute_sha256(dest)
    print(f"\nDownload complete.")
    print(f"SHA-256: {sha}")
    print(f"\nSet environment variable:")
    print(f"  export VGGT_MODEL_HASH={sha}")


def _download_progress(count, block_size, total_size):
    pct = min(100, int(count * block_size * 100 / max(total_size, 1)))
    mb = count * block_size / (1024 * 1024)
    sys.stdout.write(f"\r  {mb:.0f}MB downloaded ({pct}%)")
    sys.stdout.flush()


def cmd_reconstruct(args):
    """Run 3D reconstruction on a scene directory."""
    from .config import load_config
    from .security import load_model_secure
    from .inference import run_inference

    cfg = load_config()

    scene_dir = args.scene_dir
    image_dir = os.path.join(scene_dir, "images") if os.path.isdir(os.path.join(scene_dir, "images")) else scene_dir
    output_dir = args.output or os.path.join(scene_dir, "results")

    print(f"Loading model from {cfg.model.path}...")
    model = load_model_secure(cfg)

    print(f"Processing images from {image_dir}...")
    predictions = run_inference(model, image_dir, cfg, output_dir=output_dir)

    print(f"\nReconstruction complete!")
    print(f"Predictions saved to: {os.path.join(output_dir, 'predictions.npz')}")
    print(f"\nPrediction keys: {list(predictions.keys())}")

    # Quick point cloud summary
    from .inference import extract_point_cloud
    pts, _ = extract_point_cloud(predictions)
    print(f"Point cloud: {len(pts)} points")
    print(f"\nNext step: python -m vggt_secure.cli export --input {os.path.join(output_dir, 'predictions.npz')} --format stl")


def cmd_export(args):
    """Export predictions to SolidWorks format."""
    from .security import load_npz_secure
    from .inference import extract_point_cloud
    from .solidworks import export_format, export_all, SUPPORTED_FORMATS

    print(f"Loading predictions from {args.input}...")
    predictions = load_npz_secure(args.input)

    confidence = args.confidence or 50.0
    pts, colors = extract_point_cloud(predictions, confidence_pct=confidence)
    print(f"Point cloud: {len(pts)} points (confidence threshold: {confidence}%)")

    output_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(output_dir, exist_ok=True)

    if args.format == "all":
        print("Exporting all formats...")
        files = export_all(pts, colors, output_dir, mesh_quality=args.quality)
    else:
        print(f"Exporting {args.format.upper()}...")
        files = [export_format(args.format, pts, colors, output_dir, mesh_quality=args.quality)]

    print(f"\n{'='*60}")
    print("  EXPORTED FILES")
    print(f"{'='*60}")
    for f in files:
        print(f"  {f}")
    print(f"{'='*60}")
    _print_import_instructions(files)


def _print_import_instructions(files):
    print("\n  SOLIDWORKS IMPORT INSTRUCTIONS:")
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".stl":
            print(f"  {ext}: File > Open > select file > Import as Solid/Surface Body")
        elif ext == ".obj":
            print(f"  {ext}: Enable ScanTo3D add-in > File > Open > select file")
        elif ext == ".ply":
            print(f"  {ext}: Enable ScanTo3D > Tools > ScanTo3D > Mesh Prep Wizard")
        elif ext in (".step", ".stp"):
            print(f"  {ext}: File > Open > select file (best native CAD support)")
        elif ext in (".igs", ".iges"):
            print(f"  {ext}: File > Open > select file (legacy exchange format)")
    print()


def cmd_serve(args):
    """Start the API server."""
    from .server import run_server
    run_server()


def cmd_audit(args):
    """Print security posture summary."""
    import torch
    from .config import load_config
    from .security import compute_sha256

    cfg = load_config()

    print("VGGT-Secure Security Audit")
    print("=" * 50)

    # Model
    if os.path.isfile(cfg.model.path):
        sha = compute_sha256(cfg.model.path)
        hash_match = "VERIFIED" if cfg.model.expected_hash and sha == cfg.model.expected_hash else \
                     "NOT CONFIGURED" if not cfg.model.expected_hash else "MISMATCH"
        print(f"  Model file:      {cfg.model.path}")
        print(f"  Model hash:      {sha[:24]}...")
        print(f"  Hash check:      {hash_match}")
    else:
        print(f"  Model file:      NOT FOUND ({cfg.model.path})")

    # PyTorch
    pt_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    safe_load = "YES" if pt_ver >= (2, 6) else "NO (upgrade to 2.6+)"
    print(f"  PyTorch:         {torch.__version__}")
    print(f"  weights_only:    {safe_load}")
    print(f"  CUDA:            {'available' if torch.cuda.is_available() else 'not available'}")

    # Auth
    auth = "ENABLED" if cfg.api_key else "DISABLED"
    print(f"  API auth:        {auth}")
    print(f"  Rate limit:      {cfg.server.rate_limit_rpm} rpm")
    print(f"  Bind address:    {cfg.server.host}:{cfg.server.port}")
    local_only = cfg.server.host in ("127.0.0.1", "localhost", "::1")
    print(f"  Local only:      {'YES' if local_only else 'NO — EXPOSED TO NETWORK'}")

    # Limits
    print(f"  Max images:      {cfg.limits.max_images}")
    print(f"  Max resolution:  {cfg.limits.max_resolution}px")
    print(f"  Max upload:      {cfg.limits.max_upload_total_mb}MB")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        prog="vggt-secure",
        description="VGGT-Secure: Hardened 3D Reconstruction CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # download-model
    sub.add_parser("download-model", help="Download and verify model weights")

    # reconstruct
    p_recon = sub.add_parser("reconstruct", help="Run 3D reconstruction")
    p_recon.add_argument("--scene_dir", required=True, help="Directory with images/ subdirectory")
    p_recon.add_argument("--output", help="Output directory (default: scene_dir/results/)")

    # export
    p_export = sub.add_parser("export", help="Export to SolidWorks format")
    p_export.add_argument("--input", "-i", required=True, help="Path to predictions.npz")
    p_export.add_argument("--format", "-f", default="stl", choices=["stl", "obj", "ply", "step", "iges", "all"])
    p_export.add_argument("--output_dir", "-o", help="Output directory")
    p_export.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    p_export.add_argument("--confidence", type=float, default=50.0, help="Confidence threshold %%")

    # serve
    sub.add_parser("serve", help="Start the REST API server")

    # audit
    sub.add_parser("audit", help="Print security posture summary")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "download-model": cmd_download_model,
        "reconstruct": cmd_reconstruct,
        "export": cmd_export,
        "serve": cmd_serve,
        "audit": cmd_audit,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
