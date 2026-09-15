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


def get_current() -> tuple[Any, int] | None:
    """Read the (recorder, handle) currently open in this Task, or None.

    For callers that need to explicitly carry the binding across a
    boundary the ContextVar can't cross on its own -- e.g. resolving it on
    the CLI's own thread/task before handing off to a Rust rayon thread
    pool via `functools.partial` (see pipeline_bridge.py). Ordinary call
    sites should use `record_provider_call`/`record_internal_error`
    instead of reading this directly.
    """
    return _current.get()


def bind_current(recorder: Any | None, handle: int) -> None:
    """Explicitly set (recorder, handle) as "current" on this OS thread.

    For callers that can't rely on ContextVar auto-propagation -- a Rust
    rayon worker thread invoking a Python embed callback gets a fresh,
    empty context (contextvars don't cross OS thread boundaries the way
    they cross asyncio Task boundaries). Call this once, on that thread,
    immediately before the instrumented provider call it's meant to cover.
    `recorder=None` clears any stale binding rather than setting a
    (None, handle) pair that would itself need a None-check everywhere.
    """
    _current.set((recorder, handle) if recorder is not None else None)


def build_recorder(config: AnalyticsConfig | None, target_dir: Path) -> Any:
    """Construct the process's AnalyticsRecorder from validated config.

    Always returns a usable recorder object, even on failure -- a disabled
    recorder (every method a no-op) is the safe fallback so a bad analytics
    config can never prevent the host command from starting. `config=None`
    (e.g. a caller/test double whose Config-like object has no `analytics`
    attribute at all) is treated the same as a disabled config.
    """
    if config is None:
        return chunkhound_native.AnalyticsRecorder({"enabled": False})
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
    recorder: Any | None, command: str, source: str, action: dict[str, Any]
) -> int:
    """Open a command, set it as "current" for this Task, and return its handle.

    `recorder=None` (no recorder was wired for this call site) is a silent
    no-op returning handle 0 -- callers never need to guard this call with
    an `if recorder is not None`.

    Callers (MCP/CLI hooks) should still hold the returned handle explicitly
    and pass it to `end_command` -- the ContextVar is only a convenience for
    deeper call sites, not a replacement for that.
    """
    if recorder is None:
        return 0
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


def update_action(action: dict[str, Any]) -> None:
    """Merge new fields into whatever command is currently open in this
    Task's action, for fields only known after the command runs (e.g.
    `index`'s `file_count`/`total_chunks`, unavailable at `start_command`
    time). Reads the "current" handle the same way `record_provider_call`
    does -- for calling from deep inside a command's own implementation,
    not from the CLI/MCP dispatch chokepoint itself. A silent no-op if no
    command is open."""
    current = _current.get()
    if current is None:
        return
    recorder, handle = current
    try:
        recorder.update_action(handle, json.dumps(action, default=str))
    except Exception:
        logger.opt(exception=True).debug("analytics: update_action failed")


def end_command(recorder: Any | None, handle: int, success: bool) -> None:
    """Finalize a command and clear it as "current" for this Task.
    `recorder=None` is a silent no-op, matching `start_command`."""
    if recorder is None:
        return
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
