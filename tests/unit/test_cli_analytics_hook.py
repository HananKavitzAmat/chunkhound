"""Contract tests for the CLI dispatch analytics hook (async_main()).

Mirrors tests/unit/test_mcp_analytics_hook.py for the CLI side. Follows the
monkeypatch-create_parser/create_validated_config pattern already used in
tests/unit/test_cli_main_config_messages.py to avoid needing a real
embedding/LLM provider configured.
"""

import glob
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli import main as cli_main
from chunkhound.core.config.analytics_config import AnalyticsConfig


class _Parser:
    def __init__(self, args: SimpleNamespace) -> None:
        self._args = args

    def parse_args(self) -> SimpleNamespace:
        return self._args

    def print_help(self) -> None:
        pass


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


def _patch_common(monkeypatch: pytest.MonkeyPatch, args, tmp_path: Path) -> Path:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    config = SimpleNamespace(
        analytics=AnalyticsConfig(
            enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
        ),
        target_dir=tmp_path,
    )
    monkeypatch.setattr(
        cli_main, "create_validated_config", lambda _a, _c: (config, [])
    )
    return buffer_dir


@pytest.mark.asyncio
async def test_successful_search_command_records_one_command_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="search", verbose=False, query="explain indexing")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _ok_command)

    await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["command"] == "search"
    assert event["source"] == "cli"
    assert event["success"] is True
    assert event["action"] == {"query": "explain indexing"}


@pytest.mark.asyncio
async def test_failed_command_records_internal_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="search", verbose=False, query="x")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _failing_command)

    with pytest.raises(SystemExit):
        await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert len(events) == 1
    assert events[0]["success"] is False
    assert events[0]["internal_error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_mcp_command_is_not_wrapped_by_the_cli_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="mcp", verbose=False)
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "mcp_command", _ok_command)

    await cli_main.async_main()

    assert _read_events(buffer_dir) == []


async def _ok_command(args, config) -> None:
    return None


async def _failing_command(args, config) -> None:
    raise KeyError("boom")
