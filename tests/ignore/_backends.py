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
tests skip rust values. Those skips are allowed only for the tests listed in
``RUST_PIPELINE_SKIP_TESTS``; any other rust skip is a bug and fails the test.
"""

from __future__ import annotations

import pytest

IGNORE_BACKENDS = ("python", "libgit2", "rust", "rust-libgit2")

RUST_BACKENDS = ("rust", "rust-libgit2")

# Tests that call skip_unless_production_backend. Each entry is a test-function
# name; the skip reason names the phase that lifts it. Remove an entry when the
# pipeline starts using the Rust matcher (Phase 4 realtime, Phase 5 batch).
PHASE_4_5 = "Phase 4 (realtime) / Phase 5 (batch)"
RUST_PIPELINE_SKIP_TESTS: dict[str, str] = {
    # tests/ignore/test_ignore_matrix_new_cases.py
    "test_user_exclude_subtree_pattern_walk_uses_requested_backend": PHASE_4_5,
    "test_git_repo_batch_enumerates_with_git_ls_files": PHASE_4_5,
    "test_git_repo_batch_drops_user_exclude_subtree": PHASE_4_5,
    # tests/ignore/test_global_excludes_end_to_end.py
    "test_batch_in_git_repo_applies_glob_but_not_dir_pattern": PHASE_4_5,
    "test_batch_in_git_repo_drops_scratch_dir_like_git": PHASE_4_5,
    "test_batch_in_plain_dir_does_not_apply_core_excludes": PHASE_4_5,
    "test_realtime_admission_of_core_excludes_files_per_backend": "Phase 4 (realtime)",
    # tests/contracts/test_ignore_batch_realtime_consistency.py
    "test_batch_is_backend_independent_in_git_layouts": PHASE_4_5,
    "test_batch_and_realtime_agree_on_every_file": PHASE_4_5,
    "test_info_exclude_per_side_behavior": "Phase 4 (realtime)",
}


def ignore_mode(backend: str) -> str:
    """Semantic mode ("python" or "libgit2") that a backend value is checked against."""
    return "libgit2" if backend in ("libgit2", "rust-libgit2") else "python"


def is_rust(backend: str) -> bool:
    return backend in RUST_BACKENDS


def skip_unless_production_backend(backend: str, test_name: str) -> None:
    """Skip a rust backend only for a test on the tracked list.

    Any other test that reaches this with a rust backend fails loudly, so a
    new pipeline test cannot be silently skipped.
    """
    if not is_rust(backend):
        return
    phase = RUST_PIPELINE_SKIP_TESTS.get(test_name)
    if phase is None:
        raise AssertionError(
            f"{test_name} runs a pipeline with backend {backend!r} but is not in "
            "RUST_PIPELINE_SKIP_TESTS (tests/ignore/_backends.py); add it there "
            "with its phase, or make the test matcher-only"
        )
    pytest.skip(
        f"Rust matcher is not wired into batch/realtime pipelines until {phase}; "
        "matcher-level tests cover it"
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
