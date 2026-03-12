# CHANGES FROM UPSTREAM facebookresearch/vggt

## Files REMOVED (security/privacy reasons)

| File | Reason |
|---|---|
| `demo_gradio.py` | Replaced by authenticated FastAPI server. Gradio exposes: public tunnel (`share=True`), no auth, no rate limiting, stack traces to users (`show_error=True`), unbounded GPU allocation, user-controlled path construction, no input validation. |
| `demo_viser.py` | Binds to `0.0.0.0` with no auth. Replaced by CLI-only visualization or export-then-view workflow. If needed, run behind VPN with `--host 127.0.0.1`. |
| `requirements_demo.txt` | Contained Gradio and viser dependencies. Replaced by `requirements.txt` with pinned versions. |
| `visual_util.py` | Contained `import gradio`, `import requests` (outbound HTTP from processing code), and `download_file_from_url()`. Functionality replaced by `vggt_secure/solidworks.py` (export) and headless processing. Sky segmentation model download removed. |
| `examples/` directory | Example videos/images for the Gradio demo. Not needed for production. |
| `docs/` directory | Package installation docs for the Gradio demo variant. Replaced by this README. |
| `training/` directory | Training code — not needed for inference deployment. Keep separately if fine-tuning. |

## Files MODIFIED (security patches applied)

| Change | Original Issue | Fix |
|---|---|---|
| Model loading | `torch.hub.load_state_dict_from_url()` — downloads at runtime, no integrity check, pickle deserialization | Offline-first: download once via CLI, verify SHA-256, use `weights_only=True` (PyTorch 2.6+) |
| `np.load` calls | `allow_pickle=True` — arbitrary code execution vector | All loads use `allow_pickle=False` |
| Temp directories | Timestamp-based names, no cleanup, predictable paths | UUID4 names, auto-cleanup TTL, `mode=0o700` |
| Output filenames | User-controlled strings concatenated into file paths | Parameter-hash based filenames |
| Error handling | `show_error=True` leaks stack traces, file paths, versions | Generic error responses; internal logging only |
| Dependencies | No version pinning — supply chain attack vector | All pinned to exact versions |
| `sys.path.append()` | Import path manipulation hack | Proper package structure |
| `@spaces.GPU` decorator | HuggingFace Spaces-specific, not needed | Removed |
| Image handling | No size/count/dimension limits — GPU DoS | Configurable limits enforced before processing |
| Network exposure | `share=True`, `0.0.0.0` binding | `127.0.0.1` default, API key auth, rate limiting |

## Files ADDED

| File | Purpose |
|---|---|
| `vggt_secure/config.py` | Centralized configuration via environment variables |
| `vggt_secure/security.py` | Model integrity, input validation, rate limiting, path sanitization |
| `vggt_secure/inference.py` | Hardened inference pipeline with resource management |
| `vggt_secure/solidworks.py` | SolidWorks export (STL, OBJ, PLY, STEP, IGES) |
| `vggt_secure/server.py` | FastAPI REST API replacing Gradio |
| `vggt_secure/cli.py` | Command-line tool for local batch processing |
| `config/settings.yaml` | Reference configuration file |
| `Dockerfile` | Production container with non-root user, read-only FS |
| `CHANGES.md` | This file |

## Data Collection / Telemetry Removed

The original codebase did not contain explicit telemetry, but several patterns could leak data:

1. **`share=True` Gradio tunnel** — routes traffic through Gradio's servers, exposing uploaded images and IP addresses to a third party. **Removed.**
2. **`torch.hub.load_state_dict_from_url()`** — contacts HuggingFace CDN on every cold start, leaking server IP and request timing. **Replaced with offline loading.**
3. **`import requests` in `visual_util.py`** — `download_file_from_url()` downloads a sky segmentation model at runtime. **Removed; no runtime downloads.**
4. **`show_error=True`** — exposes internal file paths, library versions, and GPU info to any user. **Removed.**
5. **HuggingFace Hub SDK** — the `from_pretrained()` method contacts the HF API. **Replaced with direct file loading.**
