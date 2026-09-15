"""Per-user usage analytics configuration for ChunkHound.

Off by default (opt-in). See src/AGENTS.md and the ChunkHound Per-User
Analytics design doc for the full rationale — the recorder itself lives in
the Rust extension (`chunkhound_native.AnalyticsRecorder`); this module only
handles config precedence (CLI/env/file/defaults), matching every other
sub-config here.

Deliberately excluded from this model: the S3 write credential. It is read
directly from the standard AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
environment variables (see chunkhound/core/analytics/recorder.py), never as
a pydantic field here, so it can never end up persisted in a
.chunkhound.json file even by accident.
"""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyticsConfig(BaseModel):
    """Per-user usage analytics configuration.

    Configuration can be provided via:
    - Environment variables (CHUNKHOUND_ANALYTICS__*)
    - Configuration files
    - Default values
    """

    enabled: bool = Field(
        default=False,
        description="Opt-in: enable usage analytics reporting",
    )

    privacy_mode: Literal["full", "hashed", "anonymous"] = Field(
        default="full",
        description=(
            "How to represent user identity: full (OS username), "
            "hashed (salted one-way hash, local salt), or anonymous (omitted)"
        ),
    )

    s3_endpoint_url: str | None = Field(
        default=None,
        description="MinIO/S3-compatible endpoint URL to upload usage batches to",
    )

    s3_bucket: str | None = Field(
        default=None,
        description="Target bucket name for usage batch objects",
    )

    flush_interval_seconds: int = Field(
        default=21600,
        ge=1,
        description="Max time between flush attempts (default: 6 hours)",
    )

    flush_batch_size: int = Field(
        default=500,
        ge=1,
        description="Safety cap: early flush if buffered lines exceed this",
    )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load analytics config from environment variables."""
        config: dict[str, Any] = {}

        if enabled := os.getenv("CHUNKHOUND_ANALYTICS__ENABLED"):
            config["enabled"] = enabled.lower() in ("true", "1", "yes")

        if privacy_mode := os.getenv("CHUNKHOUND_ANALYTICS__PRIVACY_MODE"):
            config["privacy_mode"] = privacy_mode.strip().lower()

        if endpoint := os.getenv("CHUNKHOUND_ANALYTICS__S3_ENDPOINT_URL"):
            config["s3_endpoint_url"] = endpoint

        if bucket := os.getenv("CHUNKHOUND_ANALYTICS__S3_BUCKET"):
            config["s3_bucket"] = bucket

        if flush_interval := os.getenv("CHUNKHOUND_ANALYTICS__FLUSH_INTERVAL_SECONDS"):
            try:
                config["flush_interval_seconds"] = int(flush_interval)
            except ValueError:
                pass

        if flush_batch := os.getenv("CHUNKHOUND_ANALYTICS__FLUSH_BATCH_SIZE"):
            try:
                config["flush_batch_size"] = int(flush_batch)
            except ValueError:
                pass

        return config

    def __repr__(self) -> str:
        """String representation of analytics configuration."""
        return (
            f"AnalyticsConfig(enabled={self.enabled}, "
            f"privacy_mode={self.privacy_mode})"
        )
