# VGGT-Secure

Hardened fork of [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
for production use. Fully self-contained Docker image — one build command, no
external steps.

## What Happens When You Build

`docker build` does everything automatically:

1. Pulls NVIDIA CUDA 12.4 base image
2. Installs PyTorch 2.6 with CUDA support
3. Clones upstream `facebookresearch/vggt` and installs it
4. **Removes** Gradio, viser, training code, example assets, runtime HTTP calls
5. Installs our hardened FastAPI server, security layer, and SolidWorks exporter
6. **Downloads the 5GB model weights** and bakes them into the image
7. Computes SHA-256 hash of the model and stores it for runtime verification
8. Creates non-root user, strips SUID bits, sets file permissions
9. Produces a ready-to-run image

## Quick Start

```bash
# Build (downloads everything — takes ~15-30 min first time)
docker build -t vggt-secure .

# Run the API server
docker run --gpus all -p 8000:8000 -e VGGT_API_KEY=mysecret vggt-secure

# Test
curl http://localhost:8000/api/v1/health
```

That's it. No model downloads, no `.env` files, no volume mounts required.

## Usage

### API Server

```bash
# Start
docker run --gpus all -p 8000:8000 -e VGGT_API_KEY=mysecret vggt-secure

# Reconstruct from images
curl -X POST http://localhost:8000/api/v1/reconstruct \
  -H "Authorization: Bearer mysecret" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"
# Returns: { "job_id": "abc123", "num_points": 50000, ... }

# Export to SolidWorks STL
curl -X POST "http://localhost:8000/api/v1/export/stl?job_id=abc123" \
  -H "Authorization: Bearer mysecret" \
  --output model.stl

# Export formats: stl, obj, ply, step, iges
```

### CLI (batch processing)

```bash
# Reconstruct a directory of images
docker run --rm --gpus all \
  -v /path/to/scene:/data/scene:ro \
  -v /path/to/output:/data/output \
  vggt-secure reconstruct --scene_dir /data/scene --output /data/output

# Export predictions to SolidWorks STL
docker run --rm \
  -v /path/to/output:/data:ro \
  -v /path/to/exports:/out \
  vggt-secure export -i /data/predictions.npz -f stl -o /out

# Export all formats at once
docker run --rm \
  -v /path/to/output:/data:ro \
  -v /path/to/exports:/out \
  vggt-secure export -i /data/predictions.npz -f all -o /out

# Security audit
docker run --rm vggt-secure audit
```

### Docker Compose

```bash
cp .env.example .env
# Edit .env — set VGGT_API_KEY (generate: openssl rand -hex 32)

docker compose up -d        # Start
docker compose logs -f      # Logs
docker compose down          # Stop
```

### Makefile Shortcuts

```bash
make build                   # Build image
make run                     # Start with docker compose
make run-direct VGGT_API_KEY=key  # Run without compose
make stop                    # Stop
make logs                    # Tail logs
make audit                   # Security summary

make reconstruct SCENE=/path/to/images OUTPUT=/path/to/results
make export INPUT=/path/to/predictions.npz FMT=stl OUTPUT=./exports
```

## Production Hardening

The `docker-compose.yml` applies all of these by default:

```yaml
read_only: true                            # Immutable filesystem
tmpfs: /tmp/vggt-secure (noexec, nosuid)   # Scratch space
security_opt: no-new-privileges            # No privilege escalation
cap_drop: ALL                              # Zero Linux capabilities
memory: 32G                                # Resource ceiling
healthcheck: /api/v1/health                # Load balancer ready
```

The image also:
- Runs as non-root user `vggt`
- Has SUID/SGID bits stripped from all binaries
- Uses `weights_only=True` for model loading (blocks pickle RCE)
- Verifies model SHA-256 hash at every startup
- Returns generic errors to clients (no stack traces)
- Enforces API key auth + rate limiting on all endpoints
- Validates every uploaded image (extension, magic bytes, dimensions, file size)
- Auto-cleans temp directories on a TTL

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | No | Health check |
| `POST` | `/api/v1/reconstruct` | Yes | Upload images → 3D reconstruction |
| `POST` | `/api/v1/export/{fmt}?job_id=ID` | Yes | Export to STL/OBJ/PLY/STEP/IGES |

## SolidWorks Import

| Format | How to Import |
|---|---|
| **STL** | File → Open → select .stl |
| **OBJ** | ScanTo3D add-in → File → Open |
| **PLY** | ScanTo3D → Mesh Prep Wizard |
| **STEP** | File → Open (best for CAD editing) |
| **IGES** | File → Open (legacy) |

## Build Args

Override at build time for different configurations:

```bash
# Use commercial checkpoint (requires HF access approval)
docker build \
  --build-arg MODEL_URL=https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/model.pt \
  -t vggt-secure .

# Pin to a known model hash (build fails if mismatch)
docker build \
  --build-arg MODEL_EXPECTED_HASH=abc123... \
  -t vggt-secure .
```

## What Changed From Upstream

See [CHANGES.md](CHANGES.md). Summary: removed Gradio, viser, training code,
runtime downloads, `allow_pickle=True`, `show_error=True`, `share=True`.
Added FastAPI with auth, input validation, model integrity checks, SolidWorks export.
