"""Shared ``backend`` fixture for ignore-engine tests.

Imported by tests/ignore/conftest.py and by tests/contracts/ so every
suite that parametrizes over gitignore backends uses the same list and the
same skip rules.

Backend values and the semantics they are checked against:

- ``python``       -> Python engine, python mode (RepoAwareIgnoreEvaluator).
- ``libgit2``      -> Python engine, libgit2 mode (pygit2 required).
- ``rust``         -> native ``IgnoreMatcher`` in python mode. Checked against
                      the python-mode expectations.
- ``rust-libgit2`` -> native ``IgnoreMatcher`` in libgit2 mode (no pygit2
                      needed; the Rust code emulates the mode). Checked against
                      the libgit2-mode expectations.

``rust*`` values exercise only the matcher. Batch discovery and realtime
filtering do not use the Rust matcher until Phases 4-5, so pipeline-level
helpers call ``skip_unless_production_backend`` and skip rust values there.
"""

from __future__ import annotations

import pytest

IGNORE_BACKENDS = ("python", "libgit2", "rust", "rust-libgit2")

RUST_BACKENDS = ("rust", "rust-libgit2")


def ignore_mode(backend: str) -> str:
    """Semantic mode ("python" or "libgit2") that a backend value is checked against."""
    return "libgit2" if backend in ("libgit2", "rust-libgit2") else "python"


def is_rust(backend: str) -> bool:
    return backend in RUST_BACKENDS


def skip_unless_production_backend(backend: str) -> None:
    if is_rust(backend):
        pytest.skip(
            "Rust matcher is not wired into batch/realtime pipelines until "
            "Phases 4-5; matcher-level tests cover it"
        )


def skip_if_backend_unavailable(name: str) -> None:
    if is_rust(name):
        try:
            from chunkhound_native import IgnoreMatcher  # noqa: F401
        except (ImportError, AttributeError):
            pytest.skip("native module lacks IgnoreMatcher (build extension first)")
    if name == "libgit2":
        try:
            import pygit2  # noqa: F401
        except Exception:
            pytest.skip("pygit2 not available (libgit2 backend)")


@pytest.fixture(params=IGNORE_BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    skip_if_backend_unavailable(name)
    return name
