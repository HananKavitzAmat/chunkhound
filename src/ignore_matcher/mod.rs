//! Ignore matcher: plain-Rust core (`matcher`) plus the PyO3 class (`py`).

pub mod matcher;
pub mod py;

pub use py::PyIgnoreMatcher;
