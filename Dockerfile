# VGGT-Secure Production Dockerfile
# Multi-stage build, non-root user, minimal attack surface

# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgomp1 libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    # Remove unnecessary SUID binaries
    find / -perm /4000 -exec chmod u-s {} + 2>/dev/null || true

# Copy installed packages
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r vggt && useradd -r -g vggt -d /home/vggt -s /bin/false vggt && \
    mkdir -p /home/vggt/.cache/vggt /tmp/vggt-secure && \
    chown -R vggt:vggt /home/vggt /tmp/vggt-secure

# Copy application code
WORKDIR /app
COPY vggt_secure/ ./vggt_secure/
COPY config/ ./config/

# The upstream vggt package should be installed via requirements.txt
# or copied in if using a local checkout:
# COPY vggt/ ./vggt/

# Drop privileges
USER vggt

# Model weights should be mounted as a volume:
#   docker run -v /path/to/model.pt:/home/vggt/.cache/vggt/model.pt ...
VOLUME ["/home/vggt/.cache/vggt"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')" || exit 1

EXPOSE 8000

# Read-only filesystem (outputs go to tmpfs)
# Run with: docker run --read-only --tmpfs /tmp/vggt-secure:rw,noexec,nosuid

ENTRYPOINT ["python3.11", "-m", "vggt_secure.cli"]
CMD ["serve"]
