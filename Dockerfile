# ==========================================================================
# VGGT-Secure — Fully Self-Contained Dockerfile
#
# Everything executes inside the build. No external steps needed.
#
#   docker build -t vggt-secure .
#   docker run --gpus all -p 8000:8000 -e VGGT_API_KEY=mysecret vggt-secure
#
# That's it. Model weights, dependencies, upstream code, security patches —
# all baked into the image at build time.
#
# ==========================================================================


# ── ARGs (override at build time with --build-arg) ─────────────────────
# Which model checkpoint to bake in
ARG MODEL_URL="https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
# Set to a known hash to verify at build time. Empty = print hash only.
ARG MODEL_EXPECTED_HASH=""
# PyTorch CUDA index
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu124"


# ══════════════════════════════════════════════════════════════════════════
# Stage 1: Builder — compile native extensions, download everything
# ══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG MODEL_URL
ARG MODEL_EXPECTED_HASH
ARG TORCH_INDEX

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential git wget curl ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — everything goes here, copied cleanly to runtime stage
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch (large layer, cached unless TORCH_INDEX changes) ───────────
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url ${TORCH_INDEX}

# ── Upstream VGGT from source ──────────────────────────────────────────
RUN git clone --depth 1 https://github.com/facebookresearch/vggt.git /opt/vggt \
    && cd /opt/vggt \
    && pip install --no-cache-dir . \
    # ── Strip attack surface ───────────────────────────────────────────
    # Gradio demo — replaced by our authenticated FastAPI server
    && rm -f demo_gradio.py \
    # Viser demo — binds 0.0.0.0 unauthenticated
    && rm -f demo_viser.py \
    # visual_util.py — imports gradio + requests (outbound HTTP)
    && rm -f visual_util.py \
    # Training code — not needed for inference
    && rm -rf training/ \
    # Example assets for Gradio demo
    && rm -rf examples/ \
    # Docs for Gradio-based setup
    && rm -rf docs/ \
    # Gradio-specific requirements
    && rm -f requirements_demo.txt

# ── Our runtime deps (no Gradio, no viser) ─────────────────────────────
COPY requirements-docker.txt /tmp/reqs.txt
RUN pip install --no-cache-dir -r /tmp/reqs.txt

# ── Download model weights at build time ───────────────────────────────
RUN mkdir -p /opt/model \
    && echo "Downloading model from ${MODEL_URL} ..." \
    && wget --progress=dot:giga -O /opt/model/model.pt "${MODEL_URL}" \
    && echo "" \
    # Compute and store the SHA-256 hash inside the image
    && sha256sum /opt/model/model.pt | tee /opt/model/model.sha256 \
    && MODEL_HASH=$(sha256sum /opt/model/model.pt | cut -d' ' -f1) \
    && echo "MODEL_SHA256=${MODEL_HASH}" > /opt/model/model.env \
    # Verify against expected hash if one was provided
    && if [ -n "${MODEL_EXPECTED_HASH}" ]; then \
         ACTUAL=$(sha256sum /opt/model/model.pt | cut -d' ' -f1); \
         if [ "${ACTUAL}" != "${MODEL_EXPECTED_HASH}" ]; then \
           echo "FATAL: Model hash mismatch!"; \
           echo "  Expected: ${MODEL_EXPECTED_HASH}"; \
           echo "  Got:      ${ACTUAL}"; \
           exit 1; \
         fi; \
         echo "Model hash verified: ${ACTUAL}"; \
       else \
         echo "No expected hash provided. Hash printed above — use it."; \
       fi


# ══════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime — minimal image with everything baked in
# ══════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Minimal runtime libraries only
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgl1-mesa-glx libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # Strip SUID/SGID bits — defense in depth
    && find / -perm /4000 -type f -exec chmod u-s {} + 2>/dev/null || true \
    && find / -perm /2000 -type f -exec chmod g-s {} + 2>/dev/null || true

# Copy venv with all packages, upstream vggt code, and model weights
COPY --from=builder /opt/venv  /opt/venv
COPY --from=builder /opt/vggt  /opt/vggt
COPY --from=builder /opt/model /opt/model

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/vggt:${PYTHONPATH}"

# Non-root user
RUN groupadd -r vggt \
    && useradd -r -g vggt -d /home/vggt -m -s /usr/sbin/nologin vggt \
    && mkdir -p /tmp/vggt-secure /data \
    && chown -R vggt:vggt /home/vggt /tmp/vggt-secure /data \
    # Model is read-only for the runtime user
    && chmod -R a-w /opt/model

# Copy our hardened application
WORKDIR /app
COPY vggt_secure/ ./vggt_secure/
COPY config/      ./config/
RUN chown -R vggt:vggt /app

# Drop to non-root
USER vggt

# ── Runtime environment defaults ───────────────────────────────────────
# Model path: baked in at /opt/model/model.pt — no volume mount needed
ENV VGGT_MODEL_PATH="/opt/model/model.pt" \
    # Read the baked-in hash so the loader verifies at startup
    VGGT_BIND_HOST="0.0.0.0" \
    VGGT_BIND_PORT="8000" \
    VGGT_TEMP_DIR="/tmp/vggt-secure" \
    VGGT_LOG_LEVEL="INFO" \
    VGGT_TEMP_DIR_TTL="3600"

# Bake the model hash into the image as the default verification hash.
# The RUN reads the hash file created during build and exports it.
# If the user overrides VGGT_MODEL_HASH at runtime it takes precedence.
RUN HASH=$(cut -d' ' -f1 /opt/model/model.sha256) \
    && echo "VGGT_MODEL_HASH=${HASH}" >> /home/vggt/.env_defaults
# Shell wrapper reads this at startup (see entrypoint below).

# ── Health check ───────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

EXPOSE 8000

# ── Entrypoint ─────────────────────────────────────────────────────────
# Small shell wrapper that:
#   1. Sources the baked-in model hash if VGGT_MODEL_HASH isn't already set
#   2. Execs into the Python CLI
COPY --chown=vggt:vggt <<'ENTRYPOINT_SCRIPT' /app/entrypoint.sh
#!/bin/sh
set -e
# Load baked-in model hash if user didn't override
if [ -z "${VGGT_MODEL_HASH}" ] && [ -f /home/vggt/.env_defaults ]; then
    export $(cat /home/vggt/.env_defaults | xargs)
fi
exec python3 -m vggt_secure.cli "$@"
ENTRYPOINT_SCRIPT
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
