"""Unit tests for analytics configuration parsing."""

from pathlib import Path

import pytest

from chunkhound.core.config.analytics_config import AnalyticsConfig
from chunkhound.core.config.config import Config


def test_disabled_by_default() -> None:
    """analytics.enabled must default to False -- opt-in, per design."""
    assert AnalyticsConfig().enabled is False


def test_anonymize_defaults_to_full() -> None:
    assert AnalyticsConfig().anonymize == "full"


def test_anonymize_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        AnalyticsConfig(anonymize="incognito")


def test_save_sensitive_data_defaults_to_false() -> None:
    """Redact action content by default -- admins must explicitly opt in."""
    assert AnalyticsConfig().save_sensitive_data is False


def test_load_from_env_parses_enabled_and_anonymize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ANONYMIZE", "hashed")

    config = AnalyticsConfig.load_from_env()

    assert config["enabled"] is True
    assert config["anonymize"] == "hashed"


def test_load_from_env_parses_save_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__SAVE_SENSITIVE_DATA", "true")

    config = AnalyticsConfig.load_from_env()

    assert config["save_sensitive_data"] is True


def test_load_from_env_parses_s3_and_flush_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CHUNKHOUND_ANALYTICS__S3_ENDPOINT_URL", "https://minio.internal"
    )
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__S3_BUCKET", "usage-events")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_BATCH_SIZE", "10")

    config = AnalyticsConfig.load_from_env()

    assert config["s3_endpoint_url"] == "https://minio.internal"
    assert config["s3_bucket"] == "usage-events"
    assert config["flush_interval_seconds"] == 60
    assert config["flush_batch_size"] == 10


def test_load_from_env_ignores_malformed_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__FLUSH_INTERVAL_SECONDS", "not-a-number")

    config = AnalyticsConfig.load_from_env()

    assert "flush_interval_seconds" not in config


def test_config_composes_analytics_with_defaults(tmp_path: Path) -> None:
    """Config() must always carry a usable, disabled-by-default AnalyticsConfig,
    mirroring every other sub-config's default_factory wiring. Uses an empty
    tmp_path as target_dir so this doesn't pick up this repo's own
    .chunkhound.json (which may set its own analytics values)."""
    config = Config(target_dir=tmp_path)
    assert isinstance(config.analytics, AnalyticsConfig)
    assert config.analytics.enabled is False


def test_config_picks_up_analytics_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ANONYMIZE", "anonymous")

    config = Config(target_dir=tmp_path)

    assert config.analytics.enabled is True
    assert config.analytics.anonymize == "anonymous"
