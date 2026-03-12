# VGGT-Secure: Hardened Fork for Production Docker Deployment

Security-hardened fork of [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
with Gradio removed, FastAPI REST API, SolidWorks export, and full Docker support.

## Quick Start (Docker)

```bash
# 1. Clone and setup
git clone <this-repo> && cd vggt-secure
make setup

# 2. Configure (edit .env — set API key and model hash)
nano .env

# 3. Start the API server
make run

# 4. Test
curl http://localhost:8000/api/v1/health

# 5. Reconstruct
curl -X POST http://localhost:8000/api/v1/reconstruct \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"

# 6. Export to SolidWorks STL
curl -X POST "http://localhost:8000/api/v1/export/stl?job_id=JOB_ID_FROM_STEP_5" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  --output model.stl
```

## Step-by-Step Docker Setup

### Prerequisites
- Docker 24+ with BuildKit
- NVIDIA Container Toolkit (`nvidia-docker2`)
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)

### 1. Build the Image

```bash
docker build -t vggt-secure .
```

This clones the upstream `facebookresearch/vggt` at build time, installs it, removes
the Gradio/viser attack surface, and installs our hardened wrapper on top.

### 2. Download Model Weights

```bash
mkdir -p ./model
docker run --rm -v $(pwd)/model:/model \
  -e VGGT_MODEL_PATH=/model/model.pt \
  vggt-secure download-model
```

This downloads the ~5GB VGGT-1B weights and prints the SHA-256 hash.
Copy that hash — you'll need it for verification.

### 3. Configure Environment

```bash
cp .env.example .env

# Generate an API key
openssl rand -hex 32
# Paste into .env as VGGT_API_KEY=<key>

# Paste model hash into .env as VGGT_MODEL_HASH=<hash>
```

### 4. Run the API Server

```bash
# With docker compose (recommended)
docker compose up -d

# Or directly
docker run --gpus all \
  -p 8000:8000 \
  -e VGGT_API_KEY=your-key \
  -e VGGT_MODEL_HASH=your-hash \
  -v $(pwd)/model/model.pt:/model/model.pt:ro \
  vggt-secure
```

### 5. Run CLI Commands in Docker

```bash
# Reconstruct from a directory of images
docker run --rm --gpus all \
  -v $(pwd)/model/model.pt:/model/model.pt:ro \
  -v /path/to/scene:/data/scene:ro \
  -v /path/to/output:/data/output \
  vggt-secure reconstruct --scene_dir /data/scene --output /data/output

# Export to SolidWorks STL
docker run --rm \
  -v /path/to/output:/data:ro \
  -v /path/to/exports:/out \
  vggt-secure export -i /data/predictions.npz -f stl -o /out

# Export all formats (STL, OBJ, PLY, STEP, IGES)
docker run --rm \
  -v /path/to/output:/data:ro \
  -v /path/to/exports:/out \
  vggt-secure export -i /data/predictions.npz -f all -o /out

# Security audit
docker run --rm \
  -v $(pwd)/model/model.pt:/model/model.pt:ro \
  vggt-secure audit
```

## Makefile Shortcuts

```bash
make help              # Show all commands
make setup             # First-time: build + download model + create .env
make build             # Build Docker image
make download-model    # Download model weights
make run               # Start API server (docker compose)
make stop              # Stop API server
make logs              # Tail server logs
make audit             # Print security summary

# Reconstruct + Export
make reconstruct SCENE=/path/to/scene OUTPUT=/path/to/results
make export INPUT=/path/to/predictions.npz FMT=stl OUTPUT=/path/to/exports
```

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/health` | No | Health check for load balancers |
| POST | `/api/v1/reconstruct` | Yes | Upload images, run 3D reconstruction |
| POST | `/api/v1/export/{format}?job_id=ID` | Yes | Export to STL/OBJ/PLY/STEP/IGES |

### Example: Full Pipeline

```bash
API="http://localhost:8000"
KEY="your-api-key"

# Reconstruct
JOB=$(curl -s -X POST "$API/api/v1/reconstruct" \
  -H "Authorization: Bearer $KEY" \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job ID: $JOB"

# Export STL for SolidWorks
curl -X POST "$API/api/v1/export/stl?job_id=$JOB" \
  -H "Authorization: Bearer $KEY" \
  --output reconstruction.stl

echo "Open reconstruction.stl in SolidWorks: File > Open"
```

## Production Hardening Checklist

```bash
# Run container with all security options
docker run --gpus all \
  --read-only \
  --tmpfs /tmp/vggt-secure:rw,noexec,nosuid,size=5g \
  --security-opt no-new-privileges \
  --cap-drop ALL \
  --memory 32g \
  --cpus 8 \
  -p 8000:8000 \
  -e VGGT_API_KEY=$(cat /run/secrets/vggt_key) \
  -e VGGT_MODEL_HASH=your-sha256 \
  -v /path/to/model.pt:/model/model.pt:ro \
  vggt-secure
```

The `docker-compose.yml` applies all of these by default.

## SolidWorks Import

After exporting, import into SolidWorks:

| Format | Import Method |
|---|---|
| **STL** | File > Open > select .stl (simplest, always works) |
| **OBJ** | Enable ScanTo3D add-in > File > Open > select .obj |
| **PLY** | Enable ScanTo3D > Tools > Mesh Prep Wizard |
| **STEP** | File > Open > select .step (best for CAD editing) |
| **IGES** | File > Open > select .igs (legacy exchange) |

For best results use STEP format — it produces native B-rep geometry that SolidWorks
can edit as solid bodies. STL is the simplest fallback that always works.

## What Changed From Upstream

See [CHANGES.md](CHANGES.md) for a full diff. Summary:

**Removed:** Gradio web interface, `share=True` tunnel, `show_error=True`,
`allow_pickle=True`, unpinned dependencies, runtime model downloads, viser demo,
training code, HuggingFace Spaces code.

**Added:** FastAPI with API key auth, rate limiting, input validation (size/type/dimensions),
SHA-256 model verification, `weights_only=True` loading, UUID temp dirs with auto-cleanup,
security headers, Docker production setup, SolidWorks export pipeline.

## License

Code modifications: same license as upstream VGGT.
Model weights: use `VGGT-1B-Commercial` for commercial use (excludes military).
