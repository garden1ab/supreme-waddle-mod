# VGGT-Secure: Hardened Fork of Visual Geometry Grounded Transformer

This is a security-hardened fork of [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
for corporate/production deployment. 

## What Changed From Upstream

### Removed (Security / Privacy)
- **Gradio web interface** (`demo_gradio.py`) — replaced with authenticated FastAPI REST API
- **`share=True` public tunnel** — no public exposure by default
- **`show_error=True`** — internal stack traces never reach clients
- **`allow_pickle=True`** on `np.load` — all numpy loads use `allow_pickle=False`
- **Unpinned dependencies** — all deps locked to exact versions with hashes
- **`torch.hub.load_state_dict_from_url`** — replaced with offline-first loader + SHA-256 verification
- **Timestamp-based temp dirs** — replaced with UUID4 + auto-cleanup
- **User-controlled strings in filenames** — replaced with parameter-hash filenames
- **`visual_util.py` `import requests`** — removed external HTTP calls from processing code
- **`import gradio`** from `visual_util.py` — removed; visualization is now headless
- **`sys.path.append()`** hacks — proper package structure
- **HuggingFace Spaces `@spaces.GPU` decorator** — removed cloud-specific code

### Added (Security)
- **SHA-256 model weight verification** on every load
- **`weights_only=True`** for PyTorch 2.6+ (blocks pickle RCE)
- **Input validation** — file extension, magic bytes, dimensions, batch size, file size
- **Per-request rate limiting** via token bucket
- **API key authentication** for all endpoints
- **Request size limits** (configurable max images, max resolution, max upload MB)
- **Automatic temp directory cleanup** with configurable TTL
- **CORS lockdown** — only allowed origins
- **Security headers** — CSP, HSTS, X-Content-Type-Options, etc.
- **Structured JSON logging** — no PII, no file paths in responses
- **Resource limits** — GPU memory watchdog, request timeouts
- **SolidWorks export pipeline** — STL, OBJ, PLY, STEP, IGES conversion built-in

### Added (Functionality)
- **REST API** (`POST /api/v1/reconstruct`) — upload images, get 3D results
- **SolidWorks export** (`POST /api/v1/export/{format}`) — convert to CAD formats
- **Health check** (`GET /api/v1/health`) — for load balancer probes
- **CLI tool** (`python -m vggt_secure.cli`) — for local batch processing
- **Docker support** — production-ready Dockerfile with non-root user

## Quick Start

### Local CLI (no server needed)
```bash
# Install
pip install -r requirements.txt

# Download and verify model weights (one-time)
python -m vggt_secure.cli download-model

# Reconstruct from images
python -m vggt_secure.cli reconstruct --scene_dir /path/to/images/ --output ./results/

# Export to SolidWorks format
python -m vggt_secure.cli export --input ./results/predictions.npz --format stl --output model.stl
```

### API Server
```bash
# Set API key
export VGGT_API_KEY="your-secret-key-here"

# Start server (localhost only by default)
python -m vggt_secure.server --host 127.0.0.1 --port 8000

# With Docker
docker build -t vggt-secure .
docker run --gpus all -p 8000:8000 -e VGGT_API_KEY=your-key vggt-secure
```

### API Usage
```bash
# Reconstruct
curl -X POST http://localhost:8000/api/v1/reconstruct \
  -H "Authorization: Bearer your-key" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"

# Export to STL
curl -X POST http://localhost:8000/api/v1/export/stl \
  -H "Authorization: Bearer your-key" \
  -F "predictions=@results/predictions.npz" \
  --output model.stl
```

## Configuration

All settings via environment variables or `config/settings.yaml`:

| Variable | Default | Description |
|---|---|---|
| `VGGT_API_KEY` | (required) | API authentication key |
| `VGGT_MODEL_PATH` | `~/.cache/vggt/model.pt` | Local model weights path |
| `VGGT_MODEL_HASH` | (see config) | Expected SHA-256 of model weights |
| `VGGT_MAX_IMAGES` | `50` | Maximum images per request |
| `VGGT_MAX_RESOLUTION` | `2048` | Maximum image dimension (px) |
| `VGGT_MAX_UPLOAD_MB` | `200` | Maximum total upload size |
| `VGGT_BIND_HOST` | `127.0.0.1` | Server bind address |
| `VGGT_BIND_PORT` | `8000` | Server bind port |
| `VGGT_ALLOWED_ORIGINS` | `""` | CORS allowed origins (comma-separated) |
| `VGGT_TEMP_DIR_TTL` | `3600` | Temp directory cleanup TTL (seconds) |
| `VGGT_LOG_LEVEL` | `INFO` | Logging level |

## License

Code modifications in this fork: same license as upstream VGGT.
Model weights: use `VGGT-1B-Commercial` checkpoint for commercial use.
See upstream LICENSE.txt for full terms.
