"""
Configuration loader with environment variable overrides.
All secrets come from env vars, never from files.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    path: str = os.path.expanduser("~/.cache/vggt/model.pt")
    expected_hash: str = ""
    commercial: bool = True
    commercial_url: str = "https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/model.pt"
    original_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


@dataclass
class LimitsConfig:
    max_images: int = 50
    max_resolution: int = 2048
    max_file_size_mb: int = 50
    max_upload_total_mb: int = 200
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]
    )


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    timeout: int = 300
    max_request_size_mb: int = 250
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit_rpm: int = 30


@dataclass
class TempConfig:
    base_dir: str = "/tmp/vggt-secure"
    ttl_seconds: int = 3600
    max_disk_mb: int = 5000


@dataclass
class GpuConfig:
    memory_limit_gb: float = 0
    dtype: str = "bfloat16"


@dataclass
class AppConfig:
    api_key: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    temp: TempConfig = field(default_factory=TempConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    log_level: str = "INFO"


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    cfg = AppConfig()

    # Auth — REQUIRED
    cfg.api_key = os.environ.get("VGGT_API_KEY", "")
    if not cfg.api_key:
        logger.warning(
            "VGGT_API_KEY not set. API authentication is DISABLED. "
            "This is only acceptable for local CLI usage."
        )

    # Model
    cfg.model.path = os.path.expanduser(
        os.environ.get("VGGT_MODEL_PATH", cfg.model.path)
    )
    cfg.model.expected_hash = os.environ.get("VGGT_MODEL_HASH", "")
    cfg.model.commercial = os.environ.get("VGGT_MODEL_COMMERCIAL", "true").lower() == "true"

    # Limits
    cfg.limits.max_images = int(os.environ.get("VGGT_MAX_IMAGES", cfg.limits.max_images))
    cfg.limits.max_resolution = int(os.environ.get("VGGT_MAX_RESOLUTION", cfg.limits.max_resolution))
    cfg.limits.max_upload_total_mb = int(os.environ.get("VGGT_MAX_UPLOAD_MB", cfg.limits.max_upload_total_mb))

    # Server
    cfg.server.host = os.environ.get("VGGT_BIND_HOST", cfg.server.host)
    cfg.server.port = int(os.environ.get("VGGT_BIND_PORT", cfg.server.port))
    origins = os.environ.get("VGGT_ALLOWED_ORIGINS", "")
    if origins:
        cfg.server.allowed_origins = [o.strip() for o in origins.split(",") if o.strip()]
    cfg.server.rate_limit_rpm = int(os.environ.get("VGGT_RATE_LIMIT_RPM", cfg.server.rate_limit_rpm))

    # Temp
    cfg.temp.base_dir = os.environ.get("VGGT_TEMP_DIR", cfg.temp.base_dir)
    cfg.temp.ttl_seconds = int(os.environ.get("VGGT_TEMP_DIR_TTL", cfg.temp.ttl_seconds))

    # GPU
    cfg.gpu.dtype = os.environ.get("VGGT_DTYPE", cfg.gpu.dtype)
    cfg.gpu.memory_limit_gb = float(os.environ.get("VGGT_GPU_MEMORY_LIMIT_GB", 0))

    # Logging
    cfg.log_level = os.environ.get("VGGT_LOG_LEVEL", "INFO")

    return cfg
