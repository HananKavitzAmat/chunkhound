//! Repository-name resolution: `git remote get-url origin`, parsed to a repo
//! name, falling back to the directory basename. Never panics — a shelled-out
//! `git` that's missing, fails, or returns garbage must never break analytics
//! or the host command.

use std::path::Path;
use std::process::Command;

pub(crate) fn resolve_repository_name(dir: &Path) -> String {
    if let Some(name) = remote_repo_name(dir) {
        return name;
    }
    dir.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn remote_repo_name(dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
    parse_repo_name(&url)
}

fn parse_repo_name(url: &str) -> Option<String> {
    let trimmed = url.trim().trim_end_matches('/').trim_end_matches(".git");
    trimmed
        .rsplit(['/', ':'])
        .next()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as ProcessCommand;
    use tempfile::tempdir;

    #[test]
    fn parses_https_remote_url() {
        assert_eq!(
            parse_repo_name("https://github.com/chunkhound/chunkhound.git"),
            Some("chunkhound".to_string())
        );
    }

    #[test]
    fn parses_ssh_remote_url() {
        assert_eq!(
            parse_repo_name("git@github.com:chunkhound/chunkhound.git"),
            Some("chunkhound".to_string())
        );
    }

    #[test]
    fn falls_back_to_directory_basename_without_git() {
        let dir = tempdir().unwrap();
        let project = dir.path().join("my-project");
        std::fs::create_dir(&project).unwrap();
        assert_eq!(resolve_repository_name(&project), "my-project");
    }

    #[test]
    fn resolves_from_a_real_git_remote() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        let git = |args: &[&str]| {
            assert!(ProcessCommand::new("git")
                .args(args)
                .current_dir(repo)
                .status()
                .expect("git must be on PATH for this test")
                .success());
        };
        git(&["init", "-q"]);
        git(&[
            "remote",
            "add",
            "origin",
            "https://example.com/org/some-repo.git",
        ]);
        assert_eq!(resolve_repository_name(repo), "some-repo");
    }
}
