"""Thin Python wrapper over the native ``IgnoreMatcher`` (Phase 2, dead at runtime).

Nothing in production imports this module yet; Phases 4-5 wire it in. It
satisfies the ``_IgnoreMatcher`` protocol used by ``IndexingCoordinator``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from chunkhound_native import IgnoreMatcher as _NativeIgnoreMatcher


class RustIgnoreMatcher:
    """Native matcher with the same ``matches(path, is_dir)`` contract as Python."""

    def __init__(
        self,
        root: Path,
        sources: Iterable[str] = ("gitignore",),
        chignore_file: str = ".chignore",
        config_exclude: Iterable[str] = (),
        backend: str = "python",
        workspace_root_only_gitignore: bool | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.backend = backend
        self._native = _NativeIgnoreMatcher(
            str(self.root),
            list(sources),
            chignore_file,
            list(config_exclude),
            backend,
            workspace_root_only_gitignore,
        )

    def matches(self, path: Path, is_dir: bool = False) -> bool:
        return bool(self._native.matches(str(path), is_dir))

    def should_index(self, path: Path) -> bool:
        return bool(self._native.should_index(str(path)))
