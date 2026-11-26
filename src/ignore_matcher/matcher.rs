//! Plain-Rust ignore matcher. No PyO3 types here, so `cargo test` can run it.
//!
//! Emulates the two Python modes of `chunkhound/utils/ignore_engine.py`:
//!
//! - `Backend::Python` (`RepoAwareIgnoreEvaluator`): config excludes, then the
//!   governing `.gitignore` rules of each directory. A directory is governed by
//!   its nearest repo (`.git` dir or file) or, outside repos, by the
//!   workspace-root overlay when enabled. A nested repo (`.git` dir) is a strict
//!   boundary: parent rules do not apply to it, but a parent-ignored directory
//!   hides the repo below it. `.chignore` and `.git/info/exclude` are not read.
//! - `Backend::Libgit2` (`RepoAwareLibgit2Evaluator`), only when `gitignore` is
//!   a source: config excludes first, then the nearest repo's nested
//!   `.gitignore`, `.git/info/exclude` and `core.excludesFile`. Paths outside a
//!   repo get no gitignore filtering. The git index is not consulted.
//!
//! Config excludes are always matched against the workspace-root-relative path
//! (the Python libgit2 mode does this; the Python repo mode is handled as a
//! documented divergence, see `tests/ignore/test_ignore_matrix_new_cases.py`).
//!
//! Decisions for one path follow git: a directory that is ignored hides
//! everything below it; inside one scope the deepest `.gitignore` wins, and
//! within a file the last matching line wins.

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::Match;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

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
    /// Config excludes as given. Compiled per base directory (see `config_for`).
    config_patterns: Vec<String>,
    /// Compiled config matchers by base directory (`None` = no patterns).
    config: RwLock<HashMap<PathBuf, Option<Arc<Gitignore>>>>,
    workspace_gitignore: bool,
    global_excludes: Option<PathBuf>,
    /// Parsed ignore files keyed by path, revalidated by `(mtime, size)` on use.
    cache: RwLock<HashMap<PathBuf, CacheEntry>>,
}

struct CacheEntry {
    stamp: Option<FileStamp>,
    rules: Option<Arc<Gitignore>>,
}

/// `(mtime, size)` of a regular file. `None` for absent or non-file paths.
type FileStamp = (SystemTime, u64);

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
        let libgit2 = backend == Backend::Libgit2 && uses_gitignore(sources);
        let global = if libgit2 {
            ignore::gitignore::gitconfig_excludes_path().filter(|p| p.is_file())
        } else {
            None
        };
        Self::new_with_global_excludes(
            root,
            sources,
            config_exclude,
            backend,
            workspace_root_only,
            global,
        )
    }

    /// Same as `new`, with the `core.excludesFile` path given explicitly
    /// (libgit2 mode only; ignored in python mode).
    pub fn new_with_global_excludes(
        root: &Path,
        sources: &[String],
        config_exclude: &[String],
        backend: Backend,
        workspace_root_only: Option<bool>,
        global_excludes: Option<PathBuf>,
    ) -> Self {
        let root = canonical_or_self(root);
        let use_gitignore = uses_gitignore(sources);
        let libgit2 = backend == Backend::Libgit2 && use_gitignore;
        let config_patterns = config_exclude.to_vec();
        let workspace_gitignore = if !use_gitignore || libgit2 {
            false
        } else {
            workspace_root_only.unwrap_or_else(|| !has_repo_below(&root, &config_patterns))
        };
        Self {
            root,
            use_gitignore,
            libgit2,
            config_patterns,
            config: RwLock::new(HashMap::new()),
            workspace_gitignore,
            global_excludes: if libgit2 { global_excludes } else { None },
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// True when `path` is excluded by the active rules. Paths outside the root
    /// are never excluded (fail open).
    pub fn matches(&self, path: &Path, is_dir: bool) -> bool {
        let Some(abs) = self.resolve(path) else {
            return false;
        };
        if self.config_hides(&abs, is_dir) {
            return true;
        }

        let Some(parent) = abs.parent() else {
            return false;
        };
        let (scope, scope_is_repo) = self.scope_of(parent);

        // An ignored ancestor hides the path. Python repo mode walks every
        // ancestor (a parent rule reaches a nested repo below an ignored dir);
        // libgit2 mode asks only the nearest repo, so only ancestors inside it
        // count. A nested repo (`.git` dir) is never ignored by a parent.
        for dir in ancestors_below(&self.root, &abs) {
            if self.libgit2 && !dir.starts_with(&scope) {
                continue;
            }
            if dir.join(".git").is_dir() {
                continue;
            }
            let (dir_scope, dir_is_repo) = self.scope_of(dir.parent().unwrap_or(&self.root));
            if self.ignored_under(&dir_scope, dir_is_repo, &dir, true) {
                return true;
            }
        }

        if is_dir && abs.join(".git").is_dir() {
            return false;
        }
        self.ignored_under(&scope, scope_is_repo, &abs, is_dir)
    }

    /// Admission check for batch/realtime callers: the complement of `matches`
    /// for a path that is already on disk.
    pub fn should_index(&self, path: &Path) -> bool {
        let is_dir = path.is_dir();
        !self.matches(path, is_dir)
    }

    /// Decision for `x` using only the rules that govern it inside `scope`.
    fn ignored_under(&self, scope: &Path, scope_is_repo: bool, x: &Path, is_dir: bool) -> bool {
        if x == scope {
            return false;
        }
        let gitignore_on = if self.libgit2 {
            scope_is_repo
        } else if scope_is_repo {
            self.use_gitignore
        } else {
            self.workspace_gitignore
        };
        if !gitignore_on {
            return false;
        }
        let chain = self.chain(scope, x);
        let extras = if self.libgit2 && scope_is_repo {
            self.repo_extras(scope)
        } else {
            Vec::new()
        };
        decide(&chain, &extras, scope, x, is_dir)
    }

    /// `.gitignore` rules for every directory from `scope` down to `x`'s parent.
    fn chain(&self, scope: &Path, x: &Path) -> Vec<(PathBuf, Arc<Gitignore>)> {
        let mut out = Vec::new();
        for d in dirs_from_scope(scope, x) {
            if let Some(gi) = self.rules_for(&d.join(".gitignore"), &d) {
                out.push((d, gi));
            }
        }
        out
    }

    /// Repo-level rules that apply after nested `.gitignore` files (libgit2 only).
    fn repo_extras(&self, scope: &Path) -> Vec<Arc<Gitignore>> {
        let mut out = Vec::new();
        let info = scope.join(".git").join("info").join("exclude");
        if let Some(gi) = self.rules_for(&info, scope) {
            out.push(gi);
        }
        if let Some(global) = &self.global_excludes {
            if let Some(gi) = self.rules_for(global, &self.root) {
                out.push(gi);
            }
        }
        out
    }

    /// Config excludes. Matched against the workspace-root-relative path first
    /// (Python prunes with this spec, so it also reaches nested repos under an
    /// excluded tree). Python repo mode then matches again relative to the
    /// governing repo root; libgit2 mode uses only the workspace root.
    fn config_hides(&self, abs: &Path, is_dir: bool) -> bool {
        if self.config_patterns.is_empty() {
            return false;
        }
        if self.hit_config_base(&self.root, abs, is_dir) {
            return true;
        }
        if self.libgit2 {
            return false;
        }
        let Some(parent) = abs.parent() else {
            return false;
        };
        let (scope, scope_is_repo) = self.scope_of(parent);
        scope_is_repo && scope != self.root && self.hit_config_base(&scope, abs, is_dir)
    }

    fn hit_config_base(&self, base: &Path, abs: &Path, is_dir: bool) -> bool {
        let Some(cfg) = self.config_for(base) else {
            return false;
        };
        match abs.strip_prefix(base) {
            Ok(rel) if !rel.as_os_str().is_empty() => {
                cfg.matched_path_or_any_parents(rel, is_dir).is_ignore()
            }
            _ => false,
        }
    }

    fn config_for(&self, base: &Path) -> Option<Arc<Gitignore>> {
        if let Ok(cache) = self.config.read() {
            if let Some(hit) = cache.get(base) {
                return hit.clone();
            }
        }
        let built = build_patterns(base, &self.config_patterns).map(Arc::new);
        if let Ok(mut cache) = self.config.write() {
            cache.insert(base.to_path_buf(), built.clone());
        }
        built
    }

    /// Parsed rules for `file`, re-read when its `(mtime, size)` changes.
    fn rules_for(&self, file: &Path, base: &Path) -> Option<Arc<Gitignore>> {
        let stamp = file_stamp(file);
        if let Ok(cache) = self.cache.read() {
            if let Some(entry) = cache.get(file) {
                if entry.stamp == stamp {
                    return entry.rules.clone();
                }
            }
        }
        let rules = stamp.and_then(|_| build_file(file, base)).map(Arc::new);
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(
                file.to_path_buf(),
                CacheEntry {
                    stamp,
                    rules: rules.clone(),
                },
            );
        }
        rules
    }

    /// Nearest ancestor-or-self (within root) that holds `.git`; root otherwise.
    fn scope_of(&self, dir: &Path) -> (PathBuf, bool) {
        let mut cur = Some(dir);
        while let Some(d) = cur {
            if !d.starts_with(&self.root) {
                break;
            }
            if d.join(".git").exists() {
                return (d.to_path_buf(), true);
            }
            if d == self.root {
                break;
            }
            cur = d.parent();
        }
        (self.root.clone(), false)
    }

    /// Absolute path with symlinks resolved, as Python's `Path.resolve()` does.
    /// A missing tail is kept: the nearest existing ancestor is canonicalized and
    /// the missing components are re-appended, so deleted paths are still matched.
    fn resolve(&self, path: &Path) -> Option<PathBuf> {
        let normalized = lexical_normalize(path);
        let abs = canonicalize_lenient(&normalized)?;
        abs.starts_with(&self.root).then_some(abs)
    }
}

fn uses_gitignore(sources: &[String]) -> bool {
    sources.iter().any(|s| s == "gitignore")
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
        if dir == x {
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

/// Directories strictly between `root` and `abs`, shallowest first.
fn ancestors_below(root: &Path, abs: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut cur = abs.parent();
    while let Some(d) = cur {
        if d == root || !d.starts_with(root) {
            break;
        }
        dirs.push(d.to_path_buf());
        cur = d.parent();
    }
    dirs.reverse();
    dirs
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

fn file_stamp(file: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(file).ok()?;
    if !meta.is_file() {
        return None;
    }
    let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
    Some((mtime, meta.len()))
}

/// Parses one ignore file. Unreadable files and bad lines are skipped and
/// logged at debug level, as Python skips unreadable `.gitignore` files.
fn build_file(file: &Path, base: &Path) -> Option<Gitignore> {
    // Lossy UTF-8 decode, as Python reads .gitignore with errors="ignore": one
    // bad byte must not drop the whole file.
    let bytes = match std::fs::read(file) {
        Ok(b) => b,
        Err(err) => {
            log::debug!("ignore: cannot read {}: {err}", file.display());
            return None;
        }
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut b = GitignoreBuilder::new(base);
    for line in text.lines() {
        if let Err(err) = b.add_line(Some(file.to_path_buf()), line) {
            log::debug!("ignore: skipping line in {}: {err}", file.display());
        }
    }
    match b.build() {
        Ok(g) if !g.is_empty() => Some(g),
        Ok(_) => None,
        Err(err) => {
            log::debug!(
                "ignore: could not build rules from {}: {err}",
                file.display()
            );
            None
        }
    }
}

fn build_patterns(root: &Path, patterns: &[String]) -> Option<Gitignore> {
    if patterns.is_empty() {
        return None;
    }
    let mut b = GitignoreBuilder::new(root);
    for p in patterns {
        if let Err(err) = b.add_line(None, p) {
            log::debug!("ignore: skipping config exclude {p:?}: {err}");
        }
    }
    match b.build() {
        Ok(g) if !g.is_empty() => Some(g),
        Ok(_) => None,
        Err(err) => {
            log::debug!("ignore: could not build config excludes: {err}");
            None
        }
    }
}

/// True if any directory under `root` (root included) holds `.git`. Config
/// excluded subtrees are skipped, as the Python repo walk does.
fn has_repo_below(root: &Path, patterns: &[String]) -> bool {
    let config = build_patterns(root, patterns);
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
            if let (Some(cfg), Ok(rel)) = (config.as_ref(), child.strip_prefix(root)) {
                if cfg.matched_path_or_any_parents(rel, true).is_ignore() {
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

/// Removes `.` and resolves `..` textually, so a missing tail cannot escape a
/// parent through `..`.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn canonicalize_lenient(path: &Path) -> Option<PathBuf> {
    let mut missing: Vec<OsString> = Vec::new();
    let mut cur = path.to_path_buf();
    loop {
        if let Ok(found) = cur.canonicalize() {
            let mut out = found;
            for name in missing.iter().rev() {
                out.push(name);
            }
            return Some(out);
        }
        let name = cur.file_name()?.to_os_string();
        missing.push(name);
        cur = cur.parent()?.to_path_buf();
    }
}

fn canonical_or_self(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::Duration;

    fn write(root: &Path, rel: &str, text: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        fs::write(p, text).expect("write");
    }

    fn git_init(root: &Path) {
        fs::create_dir_all(root.join(".git")).expect("git dir");
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn m(root: &Path, backend: Backend, cfg: &[&str], srcs: &[&str]) -> IgnoreMatcher {
        IgnoreMatcher::new(root, &strings(srcs), &strings(cfg), backend, None)
    }

    const BOTH: [Backend; 2] = [Backend::Python, Backend::Libgit2];

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
        for b in BOTH {
            let mx = m(r, b, &[], &["gitignore"]);
            assert!(mx.matches(&r.join("vendor/lib/deep/s.py"), false));
            assert!(!mx.should_index(&r.join("vendor/lib/deep/s.py")));
        }
    }

    #[test]
    fn deleted_path_under_ignored_directory_is_still_ignored() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "gone/\n");
        for b in BOTH {
            let mx = m(r, b, &[], &["gitignore"]);
            assert!(mx.matches(&r.join("gone/x.py"), false));
            assert!(mx.matches(&r.join("gone/deep/y.py"), false));
            assert!(!mx.matches(&r.join("other/x.py"), false));
        }
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
    fn libgit2_mode_reads_info_exclude_and_ignores_chignore() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".git/info/exclude", "sec/\n");
        write(r, ".chignore", "*.tmp\n");
        write(r, "sec/a.key", "x");
        write(r, "pub/a.key", "x");
        write(r, "a.tmp", "x");
        let mx = m(r, Backend::Libgit2, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("sec/a.key"), false));
        assert!(!mx.matches(&r.join("pub/a.key"), false));
        assert!(!mx.matches(&r.join("a.tmp"), false));
    }

    #[test]
    fn core_excludes_file_applies_in_libgit2_mode_only() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        let global_dir = tempfile::tempdir().expect("global tmp");
        let global = global_dir.path().join("global_ignore");
        fs::write(&global, "*.secret\n").expect("global file");
        git_init(r);
        write(r, "a.secret", "x");
        write(r, "a.py", "x");
        let sources = strings(&["gitignore"]);
        let lg = IgnoreMatcher::new_with_global_excludes(
            r,
            &sources,
            &[],
            Backend::Libgit2,
            None,
            Some(global.clone()),
        );
        assert!(lg.matches(&r.join("a.secret"), false));
        assert!(!lg.matches(&r.join("a.py"), false));
        let py = IgnoreMatcher::new_with_global_excludes(
            r,
            &sources,
            &[],
            Backend::Python,
            None,
            Some(global),
        );
        assert!(!py.matches(&r.join("a.secret"), false));
    }

    #[test]
    fn core_excludes_file_applies_inside_nested_repos_in_libgit2_mode() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        let global_dir = tempfile::tempdir().expect("global tmp");
        let global = global_dir.path().join("global_ignore");
        fs::write(&global, "*.secret\n").expect("global file");
        git_init(r);
        git_init(&r.join("inner"));
        write(r, "inner/b.secret", "x");
        let lg = IgnoreMatcher::new_with_global_excludes(
            r,
            &strings(&["gitignore"]),
            &[],
            Backend::Libgit2,
            None,
            Some(global),
        );
        assert!(lg.matches(&r.join("inner/b.secret"), false));
    }

    #[test]
    fn config_excludes_apply_in_both_modes() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, "pkg/node_modules/dep/index.js", "x");
        write(r, "src/app.min.js", "x");
        write(r, "src/app.js", "x");
        for b in BOTH {
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
    fn config_excludes_apply_to_nested_repos_inside_excluded_dirs() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        git_init(&r.join("node_modules/pkg"));
        write(r, "node_modules/pkg/index.js", "x");
        for b in BOTH {
            let mx = m(r, b, &["**/node_modules/**"], &["gitignore", "config"]);
            assert!(mx.matches(&r.join("node_modules/pkg/index.js"), false));
        }
    }

    #[test]
    fn config_directory_pattern_excludes_nested_repo_and_its_files() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        git_init(&r.join("inner"));
        write(r, "inner/a.py", "x");
        for b in BOTH {
            let mx = m(r, b, &["inner/"], &["gitignore", "config"]);
            assert!(mx.matches(&r.join("inner"), true));
            assert!(mx.matches(&r.join("inner/a.py"), false));
        }
    }

    #[test]
    fn ignored_directory_containing_nested_repo_hides_it_in_python_mode() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "vendor/\n");
        git_init(&r.join("vendor/a/b"));
        write(r, "vendor/a/b/x.py", "x");
        let py = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(py.matches(&r.join("vendor/a/b/x.py"), false));
        // libgit2 mode asks only the nearest repo, so the nested repo's own
        // rules decide; the parent `vendor/` rule does not reach it.
        let lg = m(r, Backend::Libgit2, &[], &["gitignore"]);
        assert!(!lg.matches(&r.join("vendor/a/b/x.py"), false));
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
        for b in BOTH {
            let mx = m(r, b, &[], &["gitignore"]);
            assert!(mx.matches(&r.join("top.txt"), false));
            assert!(!mx.matches(&r.join("inner/a.txt"), false));
        }
    }

    #[test]
    fn edited_gitignore_is_read_again() {
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        write(r, ".gitignore", "*.log\n");
        write(r, "a.log", "x");
        write(r, "a.tmp", "x");
        let mx = m(r, Backend::Python, &[], &["gitignore"]);
        assert!(mx.matches(&r.join("a.log"), false));
        assert!(!mx.matches(&r.join("a.tmp"), false));

        // Same size, new content; the mtime is pushed forward so the change
        // is visible even on filesystems with coarse timestamps.
        write(r, ".gitignore", "*.tmp\n");
        let later = SystemTime::now() + Duration::from_secs(3600);
        fs::OpenOptions::new()
            .write(true)
            .open(r.join(".gitignore"))
            .and_then(|f| f.set_modified(later))
            .expect("set mtime");
        assert!(!mx.matches(&r.join("a.log"), false));
        assert!(mx.matches(&r.join("a.tmp"), false));
    }

    #[test]
    fn python_mode_matches_config_relative_to_nested_repo_root() {
        // Python repo mode compiles config per repo; libgit2 mode uses the
        // workspace root. Recorded in the design doc as a known difference.
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        git_init(&r.join("proj"));
        write(r, "proj/build/a.log", "x");
        write(r, "build/b.log", "x");
        let py = m(
            r,
            Backend::Python,
            &["build/*.log"],
            &["gitignore", "config"],
        );
        assert!(py.matches(&r.join("proj/build/a.log"), false));
        assert!(py.matches(&r.join("build/b.log"), false));
        let lg = m(
            r,
            Backend::Libgit2,
            &["build/*.log"],
            &["gitignore", "config"],
        );
        assert!(!lg.matches(&r.join("proj/build/a.log"), false));
        assert!(lg.matches(&r.join("build/b.log"), false));
    }

    #[test]
    fn non_utf8_gitignore_keeps_valid_lines() {
        // Python reads .gitignore with errors="ignore", so one bad byte must not
        // drop the other rules in that file.
        let t = tempfile::tempdir().expect("tmp");
        let r = t.path();
        git_init(r);
        fs::write(r.join(".gitignore"), b"\xff\xfe[\n*.tmp\n").expect("write");
        write(r, "a.tmp", "x");
        for b in BOTH {
            let mx = m(r, b, &[], &["gitignore"]);
            assert!(mx.matches(&r.join("a.tmp"), false));
        }
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
