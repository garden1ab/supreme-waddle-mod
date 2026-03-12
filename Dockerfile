# ==========================================================================
# VGGT-Secure Production Dockerfile
#
# Build:
#   docker build -t vggt-secure .
#
# Run (API server):
#   docker run --gpus all \
#     -p 8000:8000 \
#     -e VGGT_API_KEY=your-secret-key \
#     -v /path/to/model.pt:/model/model.pt:ro \
#     vggt-secure
#
# Run (CLI reconstruct):
#   docker run --gpus all \
#     -v /path/to/model.pt:/model/model.pt:ro \
#     -v /path/to/scene:/data/scene \
#     -v /path/to/output:/data/output \
#     vggt-secure reconstruct --scene_dir /data/scene --output /data/output
#
# Run (download model into host directory):
#   docker run \
#     -v /path/to/model-cache:/model \
#     vggt-secure download-model
#
# Run (export to SolidWorks STL):
#   docker run --gpus all \
#     -v /path/to/results:/data \
#     -v /path/to/output:/out \
#     vggt-secure export -i /data/predictions.npz -f stl -o /out
#
# Production hardened run:
#   docker run --gpus all \
#     --read-only \
#     --tmpfs /tmp/vggt-secure:rw,noexec,nosuid,size=5g \
#     --security-opt no-new-privileges \
#     --cap-drop ALL \
#     -p 8000:8000 \
#     -e VGGT_API_KEY=$(cat /run/secrets/vggt_key) \
#     -e VGGT_MODEL_HASH=<your-sha256> \
#     -v /path/to/model.pt:/model/model.pt:ro \
#     vggt-secure
# ==========================================================================

# ── Stage 1: Build dependencies ────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential git wget ca-certificates \
        libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch (large — separate layer for cache)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone upstream VGGT and install as package
RUN git clone --depth 1 https://github.com/facebookresearch/vggt.git /opt/vggt \
    && cd /opt/vggt \
    && pip install --no-cache-dir -e . \
    # Remove the attack surface we are replacing
    && rm -f demo_gradio.py demo_viser.py visual_util.py \
    && rm -rf training/ examples/ docs/ \
    && rm -f requirements_demo.txt

# Our pinned runtime dependencies (no Gradio, no viser)
COPY requirements-docker.txt /tmp/reqs.txt
RUN pip install --no-cache-dir -r /tmp/reqs.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgl1-mesa-glx libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && find / -perm /4000 -type f -exec chmod u-s {} + 2>/dev/null || true \
    && find / -perm /2000 -type f -exec chmod g-s {} + 2>/dev/null || true

# Portable venv + upstream vggt from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/vggt /opt/vggt

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/vggt:${PYTHONPATH}"

# Non-root user
RUN groupadd -r vggt \
    && useradd -r -g vggt -d /home/vggt -m -s /usr/sbin/nologin vggt \
    && mkdir -p /tmp/vggt-secure /data /model \
    && chown -R vggt:vggt /home/vggt /tmp/vggt-secure /data

# Our hardened application
WORKDIR /app
COPY vggt_secure/ ./vggt_secure/
COPY config/ ./config/
RUN chown -R vggt:vggt /app

USER vggt

# ── Environment defaults ───────────────────────────────────────────────
ENV VGGT_MODEL_PATH="/model/model.pt" \
    VGGT_BIND_HOST="0.0.0.0" \
    VGGT_BIND_PORT="8000" \
    VGGT_TEMP_DIR="/tmp/vggt-secure" \
    VGGT_LOG_LEVEL="INFO"

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vggt_secure.cli"]
CMD ["serve"]
