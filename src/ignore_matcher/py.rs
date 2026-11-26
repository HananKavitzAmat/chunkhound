//! PyO3 wrapper around the plain-Rust `IgnoreMatcher`. Owned `String`s cross
//! into `py.allow_threads`; no `unwrap` on the boundary.

use super::matcher::{Backend, IgnoreMatcher};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;

#[pyclass(name = "IgnoreMatcher", module = "chunkhound_native")]
pub struct PyIgnoreMatcher {
    inner: IgnoreMatcher,
}

#[pymethods]
impl PyIgnoreMatcher {
    /// `chignore_file` is accepted for signature parity with the Python engine
    /// and is intentionally unused: `.chignore` is not applied in any mode.
    #[new]
    #[pyo3(signature = (root, sources, chignore_file, config_exclude, backend, workspace_root_only_gitignore=None))]
    fn new(
        root: String,
        sources: Vec<String>,
        chignore_file: String,
        config_exclude: Vec<String>,
        backend: String,
        workspace_root_only_gitignore: Option<bool>,
    ) -> PyResult<Self> {
        let _ = chignore_file;
        let mode = Backend::parse(&backend).ok_or_else(|| {
            PyValueError::new_err(format!("unknown gitignore backend: {backend:?}"))
        })?;
        Ok(Self {
            inner: IgnoreMatcher::new(
                Path::new(&root),
                &sources,
                &config_exclude,
                mode,
                workspace_root_only_gitignore,
            ),
        })
    }

    #[pyo3(signature = (path, is_dir=false))]
    fn matches(&self, py: Python<'_>, path: String, is_dir: bool) -> bool {
        let inner = &self.inner;
        py.allow_threads(move || inner.matches(Path::new(&path), is_dir))
    }

    fn should_index(&self, py: Python<'_>, path: String) -> bool {
        let inner = &self.inner;
        py.allow_threads(move || inner.should_index(Path::new(&path)))
    }
}
