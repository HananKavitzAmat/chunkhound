"""Per-user usage analytics configuration for ChunkHound.

Off by default (opt-in). See src/AGENTS.md and the ChunkHound Per-User
Analytics design doc for the full rationale — the recorder itself lives in
the Rust extension (`chunkhound_native.AnalyticsRecorder`); this module only
handles config precedence (CLI/env/file/defaults), matching every other
sub-config here.

Deliberately excluded from this model: the S3 write credential. It is read
directly from the CHUNKHOUND_AWS_ACCESS_KEY_ID/CHUNKHOUND_AWS_SECRET_ACCESS_KEY
environment variables (see chunkhound/core/analytics/recorder.py) -- not the
standard AWS_* names, so this never silently picks up ambient AWS
credentials set for an unrelated tool -- never as a pydantic field here, so
it can never end up persisted in a .chunkhound.json file even by accident.
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

    anonymize: Literal["full", "hashed", "anonymous"] = Field(
        default="full",
        description=(
            "How to represent user identity: full (OS username), "
            "hashed (salted one-way hash, local salt), or anonymous (omitted)"
        ),
    )

    save_sensitive_data: bool = Field(
        default=False,
        description=(
            "Whether to record free-text action content (e.g. search query, "
            "research question, fetched URL) verbatim. When false (the "
            "default), those fields are still present in each event but "
            "their value is replaced with null. Independent of `anonymize`, "
            "which only controls the identity field, never action content."
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

    max_upload_retries: int = Field(
        default=10,
        ge=1,
        description=(
            "Failed-upload attempts for a buffered file before it is "
            "dropped (with a logged warning) instead of retried forever"
        ),
    )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load analytics config from environment variables."""
        config: dict[str, Any] = {}

        if enabled := os.getenv("CHUNKHOUND_ANALYTICS__ENABLED"):
            config["enabled"] = enabled.lower() in ("true", "1", "yes")

        if anonymize := os.getenv("CHUNKHOUND_ANALYTICS__ANONYMIZE"):
            config["anonymize"] = anonymize.strip().lower()

        if save_sensitive_data := os.getenv("CHUNKHOUND_ANALYTICS__SAVE_SENSITIVE_DATA"):
            config["save_sensitive_data"] = save_sensitive_data.lower() in (
                "true",
                "1",
                "yes",
            )

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

        if max_retries := os.getenv("CHUNKHOUND_ANALYTICS__MAX_UPLOAD_RETRIES"):
            try:
                config["max_upload_retries"] = int(max_retries)
            except ValueError:
                pass

        return config

    def __repr__(self) -> str:
        """String representation of analytics configuration."""
        return (
            f"AnalyticsConfig(enabled={self.enabled}, anonymize={self.anonymize}, "
            f"save_sensitive_data={self.save_sensitive_data})"
        )
