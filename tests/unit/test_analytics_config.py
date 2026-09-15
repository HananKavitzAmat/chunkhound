"""Unit tests for analytics configuration parsing."""

import pytest

from chunkhound.core.config.analytics_config import AnalyticsConfig
from chunkhound.core.config.config import Config


def test_disabled_by_default() -> None:
    """analytics.enabled must default to False -- opt-in, per design."""
    assert AnalyticsConfig().enabled is False


def test_privacy_mode_defaults_to_full() -> None:
    assert AnalyticsConfig().privacy_mode == "full"


def test_privacy_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        AnalyticsConfig(privacy_mode="incognito")


def test_load_from_env_parses_enabled_and_privacy_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__PRIVACY_MODE", "hashed")

    config = AnalyticsConfig.load_from_env()

    assert config["enabled"] is True
    assert config["privacy_mode"] == "hashed"


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


def test_config_composes_analytics_with_defaults() -> None:
    """Config() must always carry a usable, disabled-by-default AnalyticsConfig,
    mirroring every other sub-config's default_factory wiring."""
    config = Config()
    assert isinstance(config.analytics, AnalyticsConfig)
    assert config.analytics.enabled is False


def test_config_picks_up_analytics_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__ENABLED", "true")
    monkeypatch.setenv("CHUNKHOUND_ANALYTICS__PRIVACY_MODE", "anonymous")

    config = Config()

    assert config.analytics.enabled is True
    assert config.analytics.privacy_mode == "anonymous"
