//! Plain-Rust ignore matcher. No PyO3 types here, so `cargo test` can run it.
//!
//! Emulates the two Python modes of `chunkhound/utils/ignore_engine.py`:
//!
//! - `Backend::Python` (`RepoAwareIgnoreEvaluator`): config excludes, then the
//!   nearest repo's nested `.gitignore` files. Paths outside any repo use a
//!   workspace-root gitignore overlay only when it is enabled. `.chignore` is
//!   not applied and `.git/info/exclude` / `core.excludesFile` are not read.
//! - `Backend::Libgit2` (`RepoAwareLibgit2Evaluator`), only when `gitignore` is
//!   a source: config excludes first (relative to the workspace root), then the
//!   nearest repo's nested `.gitignore`, `.git/info/exclude` and
//!   `core.excludesFile`. Paths outside a repo get no gitignore filtering.
//!   The git index is not consulted.
//!
//! Decisions for one path are made the way git makes them: a directory that is
//! ignored hides everything below it; inside one level of rules the deepest
//! `.gitignore` wins, and within a file the last matching line wins.

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::Match;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Python,
    Libgit2,
}

impl Backend {
    /// Parses the `gitignore_backend` setting value. Unknown names yield `None`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "python" => Some(Self::Python),
            "libgit2" => Some(Self::Libgit2),
            _ => None,
        }
    }
}

pub struct IgnoreMatcher {
    root: PathBuf,
    use_gitignore: bool,
    libgit2: bool,
    config_patterns: Vec<String>,
    workspace_gitignore: bool,
    global_excludes: Option<PathBuf>,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    /// Parsed ignore files by file path (`None` = absent or unreadable).
    files: HashMap<PathBuf, Option<Arc<Gitignore>>>,
    /// Config-exclude matchers by scope root (`None` = no patterns).
    config: HashMap<PathBuf, Option<Arc<Gitignore>>>,
    /// Whether a directory contains a `.git` entry.
    is_repo: HashMap<PathBuf, bool>,
}

impl IgnoreMatcher {
    /// Builds a matcher. `workspace_root_only` is only consulted in python mode:
    /// `None` means "on when gitignore is a source and no repo exists under root",
    /// matching the Python default for non-repo trees.
    pub fn new(
        root: &Path,
        sources: &[String],
        config_exclude: &[String],
        backend: Backend,
        workspace_root_only: Option<bool>,
    ) -> Self {
        let root = canonical_or_self(root);
        let use_gitignore = sources.iter().any(|s| s == "gitignore");
        let libgit2 = backend == Backend::Libgit2 && use_gitignore;
        let config_patterns: Vec<String> = config_exclude.to_vec();
        let workspace_gitignore = if !use_gitignore || libgit2 {
            false
        } else {
            workspace_root_only.unwrap_or_else(|| {
                let config = build_gitignore(&root, &config_patterns);
                !has_repo_below(&root, &config)
            })
        };
        let global_excludes = if libgit2 {
            ignore::gitignore::gitconfig_excludes_path().filter(|p| p.is_file())
        } else {
            None
        };
        Self {
            root,
            use_gitignore,
            libgit2,
            config_patterns,
            workspace_gitignore,
            global_excludes,
            cache: Mutex::new(Cache::default()),
        }
    }

    /// True when `path` is excluded by the active rules. Paths outside the root
    /// are never excluded (fail open).
    pub fn matches(&self, path: &Path, is_dir: bool) -> bool {
        let Some(abs) = self.resolve(path) else {
            return false;
        };
        let mut cache = self.lock();
        let (scope, is_repo) = self.scope_for(&abs, is_dir, &mut cache);
        if abs == scope {
            // A repo or workspace root is never excluded by its own rules.
            return false;
        }

        let cfg_scope = if self.libgit2 {
            self.root.clone()
        } else {
            scope.clone()
        };
        if let Some(cfg) = self.config_for(&cfg_scope, &mut cache) {
            if let Ok(rel) = abs.strip_prefix(&cfg_scope) {
                if !rel.as_os_str().is_empty()
                    && cfg.matched_path_or_any_parents(rel, is_dir).is_ignore()
                {
                    return true;
                }
            }
        }

        let gitignore_on = if self.libgit2 {
            is_repo
        } else if is_repo {
            self.use_gitignore
        } else {
            self.workspace_gitignore
        };
        if !gitignore_on {
            return false;
        }

        let dirs = dirs_from_scope(&scope, &abs);
        let mut chain: Vec<(PathBuf, Arc<Gitignore>)> = Vec::new();
        for d in &dirs {
            if let Some(gi) = load_cached(&mut cache, &d.join(".gitignore"), d) {
                chain.push((d.clone(), gi));
            }
        }
        let extras = self.root_level_extras(&scope, is_repo, &mut cache);

        for d in dirs.iter().skip(1) {
            if decide(&chain, &extras, &scope, d, true) {
                return true;
            }
        }
        decide(&chain, &extras, &scope, &abs, is_dir)
    }

    /// Admission check for batch/realtime callers: the complement of `matches`
    /// for a path that is already on disk.
    pub fn should_index(&self, path: &Path) -> bool {
        let is_dir = path.is_dir();
        !self.matches(path, is_dir)
    }

    fn lock(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn resolve(&self, path: &Path) -> Option<PathBuf> {
        let abs = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                let parent = path.parent()?.canonicalize().ok()?;
                parent.join(path.file_name()?)
            }
        };
        abs.starts_with(&self.root).then_some(abs)
    }

    /// Nearest ancestor-or-self (within root) that holds `.git`; root otherwise.
    fn scope_for(&self, abs: &Path, is_dir: bool, cache: &mut Cache) -> (PathBuf, bool) {
        let mut cur = if is_dir { Some(abs) } else { abs.parent() };
        while let Some(d) = cur {
            if !d.starts_with(&self.root) {
                break;
            }
            if is_repo_dir(d, cache) {
                return (d.to_path_buf(), true);
            }
            if d == self.root {
                break;
            }
            cur = d.parent();
        }
        (self.root.clone(), false)
    }

    fn config_for(&self, scope: &Path, cache: &mut Cache) -> Option<Arc<Gitignore>> {
        if self.config_patterns.is_empty() {
            return None;
        }
        cache
            .config
            .entry(scope.to_path_buf())
            .or_insert_with(|| Some(Arc::new(build_gitignore(scope, &self.config_patterns))))
            .clone()
    }

    /// Repo-level rules that apply after nested `.gitignore` files (libgit2 only).
    fn root_level_extras(
        &self,
        scope: &Path,
        is_repo: bool,
        cache: &mut Cache,
    ) -> Vec<Arc<Gitignore>> {
        let mut out = Vec::new();
        if !(self.libgit2 && is_repo) {
            return out;
        }
        let git_dir = scope.join(".git");
        if git_dir.is_dir() {
            if let Some(gi) = load_cached(cache, &git_dir.join("info").join("exclude"), scope) {
                out.push(gi);
            }
        }
        if let Some(global) = &self.global_excludes {
            if let Some(gi) = load_cached(cache, global, &self.root) {
                out.push(gi);
            }
        }
        out
    }
}

/// Git precedence for one path: deepest applicable file first, then repo-level
/// rules. Only directories strictly above `x` apply to `x`.
fn decide(
    chain: &[(PathBuf, Arc<Gitignore>)],
    extras: &[Arc<Gitignore>],
    scope: &Path,
    x: &Path,
    is_dir: bool,
) -> bool {
    for (dir, gi) in chain.iter().rev() {
        if dir == x || !x.starts_with(dir) {
            continue;
        }
        let Ok(rel) = x.strip_prefix(dir) else {
            continue;
        };
        match gi.matched(rel, is_dir) {
            Match::Ignore(_) => return true,
            Match::Whitelist(_) => return false,
            Match::None => {}
        }
    }
    let Ok(rel) = x.strip_prefix(scope) else {
        return false;
    };
    for gi in extras {
        match gi.matched(rel, is_dir) {
            Match::Ignore(_) => return true,
            Match::Whitelist(_) => return false,
            Match::None => {}
        }
    }
    false
}

/// Directories from `scope` (inclusive) down to the parent of `abs`, shallowest first.
fn dirs_from_scope(scope: &Path, abs: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut cur = abs.parent();
    while let Some(d) = cur {
        if !d.starts_with(scope) {
            return Vec::new();
        }
        dirs.push(d.to_path_buf());
        if d == scope {
            break;
        }
        cur = d.parent();
    }
    dirs.reverse();
    dirs
}

fn load_cached(cache: &mut Cache, file: &Path, root: &Path) -> Option<Arc<Gitignore>> {
    cache
        .files
        .entry(file.to_path_buf())
        .or_insert_with(|| {
            if !file.is_file() {
                return None;
            }
            let mut b = GitignoreBuilder::new(root);
            let _ = b.add(file);
            b.build().ok().filter(|g| !g.is_empty()).map(Arc::new)
        })
        .clone()
}

fn is_repo_dir(dir: &Path, cache: &mut Cache) -> bool {
    *cache
        .is_repo
        .entry(dir.to_path_buf())
        .or_insert_with(|| dir.join(".git").exists())
}

fn build_gitignore(root: &Path, patterns: &[String]) -> Gitignore {
    let mut b = GitignoreBuilder::new(root);
    for p in patterns {
        let _ = b.add_line(None, p);
    }
    b.build().unwrap_or_else(|_| Gitignore::empty())
}

/// True if any directory under `root` (root included) holds `.git`. Config
/// excluded subtrees are skipped, as the Python repo walk does.
fn has_repo_below(root: &Path, config: &Gitignore) -> bool {
    if root.join(".git").exists() {
        return true;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(ft) = entry.file_type() else {
                continue;
            };
            if !ft.is_dir() || entry.file_name() == ".git" {
                continue;
            }
            let child = entry.path();
            if let Ok(rel) = child.strip_prefix(root) {
                if config.matched_path_or_any_parents(rel, true).is_ignore() {
                    continue;
                }
            }
            if child.join(".git").exists() {
                return true;
            }
            stack.push(child);
        }
    }
    false
}

fn canonical_or_self(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(root: &Path, rel: &str, text: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        fs::write(p, text).expect("write");
    }

    fn git_init(root: &Path) {
        fs::create_dir_all(root.join(".git")).expect("git dir");
    }

    fn m(root: &Path, backend: Backend, cfg: &[&str], srcs: &[&str]) -> IgnoreMatcher {
        let sources: Vec<String> = srcs.iter().map(|s| s.to_string()).collect();
        let cfg: Vec<String> = cfg.iter().map(|s| s.to_string()).collect();
        IgnoreMatcher::new(root, &sources, &cfg, backend, None)
    }

    #[test]
    fn gitignore_file_and_dir_rules() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.log\nbuild/\n");
        write(r, "a.log", "x");
        write(r, "build/x.py", "x");
        write(r, "keep.py", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("a.log"), false));
        assert!(mx.matches(&r.join("build"), true));
        assert!(!mx.matches(&r.join("keep.py"), false));
    }

    #[test]
    fn ignored_directory_hides_children() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "vendor/\n");
        write(r, "vendor/lib/deep/s.py", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("vendor/lib/deep/s.py"), false));
        assert!(!mx.should_index(&r.join("vendor/lib/deep/s.py")));
    }

    #[test]
    fn nested_gitignore_negation_and_anchoring() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.log\n");
        write(r, "sub/.gitignore", "!keep.log\n/only_here.txt\n");
        write(r, "sub/keep.log", "x");
        write(r, "sub/other.log", "x");
        write(r, "sub/only_here.txt", "x");
        write(r, "other/only_here.txt", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(!mx.matches(&r.join("sub/keep.log"), false));
        assert!(mx.matches(&r.join("sub/other.log"), false));
        assert!(mx.matches(&r.join("sub/only_here.txt"), false));
        assert!(!mx.matches(&r.join("other/only_here.txt"), false));
    }

    #[test]
    fn character_class_and_question_mark() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "[ab].log\nq?.txt\n");
        for f in ["a.log", "c.log", "q1.txt", "q.txt"] {
            write(r, f, "x");
        }
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("a.log"), false));
        assert!(!mx.matches(&r.join("c.log"), false));
        assert!(mx.matches(&r.join("q1.txt"), false));
        assert!(!mx.matches(&r.join("q.txt"), false));
    }

    #[test]
    fn python_mode_ignores_info_exclude_and_chignore() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".git/info/exclude", "sec/\n");
        write(r, ".chignore", "*.tmp\n");
        write(r, "sec/a.key", "x");
        write(r, "a.tmp", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(!mx.matches(&r.join("sec/a.key"), false));
        assert!(!mx.matches(&r.join("a.tmp"), false));
    }

    #[test]
    fn libgit2_mode_reads_info_exclude() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".git/info/exclude", "sec/\n");
        write(r, "sec/a.key", "x");
        write(r, "pub/a.key", "x");
        let mx = m(r, Backend::Libgit2, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("sec/a.key"), false));
        assert!(!mx.matches(&r.join("pub/a.key"), false));
    }

    #[test]
    fn libgit2_mode_ignores_chignore() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".chignore", "*.tmp\n");
        write(r, "a.tmp", "x");
        let mx = m(r, Backend::Libgit2, &[], &["gitignore"]);
        assert!(!mx.matches(&r.join("a.tmp"), false));
    }

    #[test]
    fn config_excludes_apply_in_both_modes() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, "pkg/node_modules/dep/index.js", "x");
        write(r, "src/app.min.js", "x");
        write(r, "src/app.js", "x");
        for b in [Backend::Python, Backend::Libgit2] {
            let mx = m(
                r,
                b,
                &["**/node_modules/**", "*.min.js"],
                &["gitignore", "config"],
            );
            assert!(mx.matches(&r.join("pkg/node_modules/dep/index.js"), false));
            assert!(mx.matches(&r.join("src/app.min.js"), false));
            assert!(!mx.matches(&r.join("src/app.js"), false));
        }
    }

    #[test]
    fn libgit2_config_excludes_fail_open_outside_repo() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        write(r, ".gitignore", "*.log\n");
        write(r, "a.log", "x");
        write(r, "b.txt", "x");
        let mx = m(r, Backend::Libgit2, &["*.txt"], &["gitignore"]);
        assert!(!mx.matches(&r.join("a.log"), false));
        assert!(mx.matches(&r.join("b.txt"), false));
    }

    #[test]
    fn non_repo_root_gitignore_applies_only_in_python_mode_default() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        write(r, ".gitignore", "*.log\n");
        write(r, "a.log", "x");
        let py = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(py.matches(&r.join("a.log"), false));
        let lg = m(r, Backend::Libgit2, &[], &["gitignore"]);
        assert!(!lg.matches(&r.join("a.log"), false));
    }

    #[test]
    fn nested_repo_is_a_strict_boundary() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.txt\n");
        git_init(&r.join("inner"));
        write(r, "inner/a.txt", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("top.txt"), false));
        assert!(!mx.matches(&r.join("inner/a.txt"), false));
    }

    #[test]
    fn gitignore_source_absent_disables_rules() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.log\n");
        write(r, "a.log", "x");
        let mx = m(r, Backend::Python, &[], &["config"]);
        assert!(!mx.matches(&r.join("a.log"), false));
    }

    #[test]
    fn paths_outside_root_fail_open() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.log\n");
        let other = tempfile::tempdir().expect("tmp2");
        write(other.path(), "x.log", "x");
        let mx = m(r, Backend::Python, &["*.log"], &["gitignore", "config"]);
        assert!(!mx.matches(&other.path().join("x.log"), false));
    }
}
