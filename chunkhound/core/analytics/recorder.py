"""Construction/lifecycle glue for the Rust-native `AnalyticsRecorder`.

This module deliberately owns no bookkeeping of its own -- that all lives in
`chunkhound_native.AnalyticsRecorder` (handle-based, see src/analytics/ and
src/AGENTS.md). What lives here:

- `build_recorder()`: turns a validated `AnalyticsConfig` + the resolved
  target directory into a constructed native recorder. Always succeeds --
  falls back to a disabled recorder on any construction failure, since
  analytics must never be the reason a host command fails to start.
- A `contextvars`-based "current (recorder, handle)" convenience, used by
  call sites that don't have an explicit handle to hand (LLM/embedding
  provider calls, several stack frames below the MCP/CLI hook that opened
  the command). This is safe for ordinary single-threaded Python call
  chains -- each asyncio Task gets its own copy of the context -- and is
  NOT used across the Rust rayon thread-pool boundary during indexing;
  that boundary threads the recorder through explicitly instead (see
  chunkhound/pipeline_bridge.py and the Phase 4 native-path wiring).
"""

import contextvars
import getpass
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

import chunkhound_native
from chunkhound import __version__
from chunkhound.core.config.analytics_config import AnalyticsConfig

_ANALYTICS_DIR = Path.home() / ".config" / "chunkhound" / "analytics"

# (recorder, handle) for the command currently open in this asyncio
# Task/thread of control, or None. Never shared across asyncio Tasks or OS
# threads -- see module docstring.
_current: "contextvars.ContextVar[tuple[Any, int] | None]" = contextvars.ContextVar(
    "chunkhound_analytics_current", default=None
)


def build_recorder(config: AnalyticsConfig, target_dir: Path) -> Any:
    """Construct the process's AnalyticsRecorder from validated config.

    Always returns a usable recorder object, even on failure -- a disabled
    recorder (every method a no-op) is the safe fallback so a bad analytics
    config can never prevent the host command from starting.
    """
    config_dict = {
        "enabled": config.enabled,
        "privacy_mode": config.privacy_mode,
        "s3_endpoint_url": config.s3_endpoint_url,
        "s3_bucket": config.s3_bucket,
        # Read directly from the standard AWS env vars, never from
        # AnalyticsConfig -- see analytics_config.py's module docstring for
        # why this must never be a pydantic field (so it can never end up
        # persisted in .chunkhound.json).
        "s3_access_key": _get_env("AWS_ACCESS_KEY_ID"),
        "s3_secret_key": _get_env("AWS_SECRET_ACCESS_KEY"),
        "flush_interval_seconds": config.flush_interval_seconds,
        "flush_batch_size": config.flush_batch_size,
        "buffer_dir": str(_ANALYTICS_DIR),
        "salt_path": str(_ANALYTICS_DIR / "salt"),
        "repository_dir": str(target_dir),
        "os_username": _get_os_username(),
        "chunkhound_version": __version__,
    }
    try:
        return chunkhound_native.AnalyticsRecorder(config_dict)
    except Exception:
        logger.opt(exception=True).debug(
            "analytics: failed to construct recorder, disabling for this process"
        )
        return chunkhound_native.AnalyticsRecorder({"enabled": False})


def start_command(
    recorder: Any, command: str, source: str, action: dict[str, Any]
) -> int:
    """Open a command, set it as "current" for this Task, and return its handle.

    Callers (MCP/CLI hooks) should still hold the returned handle explicitly
    and pass it to `end_command` -- the ContextVar is only a convenience for
    deeper call sites, not a replacement for that.
    """
    try:
        handle = int(
            recorder.start_command(command, source, json.dumps(action, default=str))
        )
    except Exception:
        logger.opt(exception=True).debug(
            "analytics: start_command failed, dropping event"
        )
        return 0
    _current.set((recorder, handle))
    return handle


def end_command(recorder: Any, handle: int, success: bool) -> None:
    """Finalize a command and clear it as "current" for this Task."""
    try:
        recorder.end_command(handle, success)
    except Exception:
        logger.opt(exception=True).debug("analytics: end_command failed")
    finally:
        _current.set(None)


def record_provider_call(
    kind: str,
    provider: str,
    model: str,
    success: bool,
    error_type: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Record one provider-call attempt against whatever command is
    currently open in this Task. A silent no-op if none is open -- matches
    the design's "shouldn't normally happen" fallback for provider calls
    made outside any tracked command."""
    current = _current.get()
    if current is None:
        return
    recorder, handle = current
    try:
        recorder.record_provider_call(
            handle,
            kind,
            provider,
            model,
            success,
            error_type,
            input_tokens,
            output_tokens,
        )
    except Exception:
        logger.opt(exception=True).debug("analytics: record_provider_call failed")


def record_internal_error(error_type: str) -> None:
    """Record a command-level failure not caused by any vendor call, against
    whatever command is currently open in this Task."""
    current = _current.get()
    if current is None:
        return
    recorder, handle = current
    try:
        recorder.record_internal_error(handle, error_type)
    except Exception:
        logger.opt(exception=True).debug("analytics: record_internal_error failed")


def _get_env(name: str) -> str | None:
    return os.environ.get(name)


def _get_os_username() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
