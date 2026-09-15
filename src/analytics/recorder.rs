//! `AnalyticsRecorder` — the PyO3-exposed core. Owns the local buffer file,
//! a background flush thread, and the S3 upload. Every public method here
//! (Python-facing or Rust-internal) must be panic-free and must never let an
//! internal I/O/config failure disrupt the caller — see `mod.rs`.

use super::command::CommandTable;
use super::identity;
use super::repository::resolve_repository_name;
use super::s3::S3Target;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::FromPyObject;
use rand::RngCore;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Config {
    flush_interval: Duration,
    flush_batch_size: usize,
    max_buffer_bytes: u64,
    buffer_dir: PathBuf,
    s3_target: Option<S3Target>,
}

struct BufferState {
    active_path: PathBuf,
    lines: usize,
    bytes: u64,
    last_flush: Instant,
}

pub(crate) struct Inner {
    config: Config,
    commands: CommandTable,
    repository: String,
    payload_identity: Option<String>,
    key_segment: String,
    /// The *Python package's* version (`chunkhound.__version__`, hatch-vcs
    /// derived), passed in from Python at construction time — deliberately
    /// not `env!("CARGO_PKG_VERSION")`, which is this crate's own fixed
    /// `Cargo.toml` version and not what the wiki's `chunkhound_version`
    /// field means.
    chunkhound_version: String,
    buffer: Mutex<BufferState>,
    http: reqwest::blocking::Client,
    shutdown: AtomicBool,
}

#[pyclass]
pub struct AnalyticsRecorder {
    /// `None` when analytics is disabled or misconfigured — every method
    /// below degrades to a silent no-op in that case, by construction.
    inner: Option<Arc<Inner>>,
    flush_thread: Mutex<Option<JoinHandle<()>>>,
}

#[pymethods]
impl AnalyticsRecorder {
    /// `config` keys: enabled (bool), privacy_mode (str), s3_endpoint_url
    /// (str|None), s3_bucket (str|None), s3_access_key (str|None),
    /// s3_secret_key (str|None), flush_interval_seconds (int),
    /// flush_batch_size (int), buffer_dir (str), salt_path (str),
    /// repository_dir (str), os_username (str), chunkhound_version (str —
    /// pass `chunkhound.__version__`, not this crate's own version).
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let inner = build_inner(config);
        let mut recorder = Self {
            inner: inner.map(Arc::new),
            flush_thread: Mutex::new(None),
        };
        recorder.spawn_flush_thread();
        Ok(recorder)
    }

    /// Returns an opaque handle id. Pass it into every subsequent call for
    /// this command. `action` must be a JSON-serialized object (built with
    /// `json.dumps` on the Python side) — a malformed payload degrades to
    /// `{}` rather than raising, since a bad action string must never break
    /// the host command.
    #[pyo3(signature = (command, source, action))]
    fn start_command(&self, command: String, source: String, action: String) -> u64 {
        let Some(inner) = &self.inner else { return 0 };
        let action_value: serde_json::Value =
            serde_json::from_str(&action).unwrap_or_else(|_| json!({}));
        inner.commands.start(command, source, action_value)
    }

    #[pyo3(signature = (handle, kind, provider, model, success, error_type=None, input_tokens=None, output_tokens=None))]
    #[allow(clippy::too_many_arguments)] // Mirrors the wiki's own record_provider_call() signature.
    fn record_provider_call(
        &self,
        handle: u64,
        kind: String,
        provider: String,
        model: String,
        success: bool,
        error_type: Option<String>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        let Some(inner) = &self.inner else { return };
        inner.commands.record_provider_call(
            handle,
            super::command::ProviderCall {
                kind: &kind,
                provider: &provider,
                model: &model,
                success,
                error_type: error_type.as_deref(),
                input_tokens,
                output_tokens,
            },
        );
    }

    fn record_internal_error(&self, handle: u64, error_type: String) {
        let Some(inner) = &self.inner else { return };
        inner.commands.record_internal_error(handle, error_type);
    }

    /// Merges `action` (a JSON-serialized object) into a still-open
    /// command's action fields -- for fields only known after the command
    /// runs (e.g. `index`'s `file_count`/`total_chunks`). See
    /// `CommandTable::update_action`.
    fn update_action(&self, handle: u64, action: String) {
        let Some(inner) = &self.inner else { return };
        if let Ok(value) = serde_json::from_str(&action) {
            inner.commands.update_action(handle, value);
        }
    }

    fn end_command(&self, handle: u64, success: bool) {
        let Some(inner) = &self.inner else { return };
        let Some(state) = inner.commands.end(handle) else {
            return;
        };
        let event = build_event(inner, &state, success);
        inner.record_event(event);
        inner.maybe_flush(false);
    }

    /// Best-effort final flush, bounded by `timeout_ms` so a slow/unreachable
    /// S3 endpoint can never hang process exit. Call from the CLI's
    /// `finally` and the MCP server's shutdown path.
    #[pyo3(signature = (timeout_ms=3000))]
    fn shutdown(&self, py: Python<'_>, timeout_ms: u64) {
        let Some(inner) = self.inner.clone() else {
            return;
        };
        inner.shutdown.store(true, Ordering::SeqCst);
        py.allow_threads(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            let flush_inner = inner.clone();
            std::thread::spawn(move || {
                flush_inner.maybe_flush(true);
                let _ = tx.send(());
            });
            let _ = rx.recv_timeout(Duration::from_millis(timeout_ms));
        });
    }
}

impl AnalyticsRecorder {
    fn spawn_flush_thread(&mut self) {
        let Some(inner) = self.inner.clone() else {
            return;
        };
        let handle = std::thread::spawn(move || {
            while !inner.shutdown.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(1));
                inner.maybe_flush(false);
            }
        });
        *self.flush_thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
    }
}

/// Plain-Rust mirror of the Python config dict — kept separate from the
/// PyO3 boundary specifically so `build_inner_from_raw` (the actual
/// construction logic) can be exercised by `cargo test` without a Python
/// interpreter/GIL involved at all.
struct RawConfig {
    enabled: bool,
    privacy_mode: String,
    s3_endpoint_url: Option<String>,
    s3_bucket: Option<String>,
    s3_access_key: Option<String>,
    s3_secret_key: Option<String>,
    flush_interval_seconds: u64,
    flush_batch_size: usize,
    max_buffer_bytes: u64,
    buffer_dir: PathBuf,
    salt_path: PathBuf,
    repository_dir: PathBuf,
    os_username: String,
    /// `chunkhound.__version__` as passed from Python. Defaulting to
    /// `"unknown"` (not this crate's own version) if never supplied, since
    /// a caller that omits it has no other correct value to fall back to.
    chunkhound_version: String,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            privacy_mode: "full".to_string(),
            s3_endpoint_url: None,
            s3_bucket: None,
            s3_access_key: None,
            s3_secret_key: None,
            flush_interval_seconds: 21600,
            flush_batch_size: 500,
            max_buffer_bytes: 10 * 1024 * 1024,
            buffer_dir: PathBuf::from("."),
            salt_path: PathBuf::from("salt"),
            repository_dir: PathBuf::from("."),
            os_username: "unknown".to_string(),
            chunkhound_version: "unknown".to_string(),
        }
    }
}

fn build_inner(config: &Bound<'_, PyDict>) -> Option<Inner> {
    let buffer_dir: String = get(config, "buffer_dir")?;
    let salt_path: String = get(config, "salt_path")?;
    let raw = RawConfig {
        enabled: get(config, "enabled").unwrap_or(false),
        privacy_mode: get(config, "privacy_mode").unwrap_or_else(|| "full".to_string()),
        s3_endpoint_url: get(config, "s3_endpoint_url"),
        s3_bucket: get(config, "s3_bucket"),
        s3_access_key: get(config, "s3_access_key"),
        s3_secret_key: get(config, "s3_secret_key"),
        flush_interval_seconds: get(config, "flush_interval_seconds").unwrap_or(21600),
        flush_batch_size: get(config, "flush_batch_size").unwrap_or(500),
        buffer_dir: PathBuf::from(buffer_dir),
        salt_path: PathBuf::from(salt_path),
        repository_dir: PathBuf::from(
            get(config, "repository_dir").unwrap_or_else(|| ".".to_string()),
        ),
        os_username: get(config, "os_username").unwrap_or_else(|| "unknown".to_string()),
        chunkhound_version: get(config, "chunkhound_version")
            .unwrap_or_else(|| "unknown".to_string()),
        ..RawConfig::default()
    };
    build_inner_from_raw(raw)
}

fn build_inner_from_raw(raw: RawConfig) -> Option<Inner> {
    if !raw.enabled {
        return None;
    }

    let s3_target = match (
        &raw.s3_endpoint_url,
        &raw.s3_bucket,
        &raw.s3_access_key,
        &raw.s3_secret_key,
    ) {
        (Some(url), Some(bucket), Some(key), Some(secret)) => {
            match S3Target::new(url, bucket, key, secret) {
                Ok(target) => Some(target),
                Err(e) => {
                    log::warn!("analytics: disabling upload, invalid S3 config: {e}");
                    None
                }
            }
        }
        _ => {
            log::debug!("analytics: no S3 target configured, buffering locally only");
            None
        }
    };

    if let Err(e) = fs::create_dir_all(&raw.buffer_dir) {
        log::warn!("analytics: disabling, cannot create buffer dir: {e}");
        return None;
    }
    let repository = resolve_repository_name(&raw.repository_dir);
    let payload_identity =
        identity::payload_identity(&raw.privacy_mode, &raw.os_username, &raw.salt_path);
    let key_segment = identity::object_key_segment(&raw.privacy_mode, &payload_identity);

    let active_path = new_active_buffer_path(&raw.buffer_dir);
    let http = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(client) => client,
        Err(e) => {
            log::warn!("analytics: disabling, failed to build HTTP client: {e}");
            return None;
        }
    };

    Some(Inner {
        config: Config {
            flush_interval: Duration::from_secs(raw.flush_interval_seconds),
            flush_batch_size: raw.flush_batch_size,
            max_buffer_bytes: raw.max_buffer_bytes,
            buffer_dir: raw.buffer_dir,
            s3_target,
        },
        commands: CommandTable::default(),
        repository,
        payload_identity,
        key_segment,
        chunkhound_version: raw.chunkhound_version,
        buffer: Mutex::new(BufferState {
            active_path,
            lines: 0,
            bytes: 0,
            last_flush: Instant::now(),
        }),
        http,
        shutdown: AtomicBool::new(false),
    })
}

fn get<'py, T: FromPyObject<'py>>(config: &Bound<'py, PyDict>, key: &str) -> Option<T> {
    config.get_item(key).ok().flatten()?.extract().ok()
}

fn new_active_buffer_path(buffer_dir: &Path) -> PathBuf {
    let pid = std::process::id();
    let start_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    buffer_dir.join(format!("buffer-{pid}-{start_ts}.jsonl"))
}

fn build_event(
    inner: &Inner,
    state: &super::command::CommandState,
    success: bool,
) -> serde_json::Value {
    let mut providers = serde_json::Map::new();
    for ((kind, provider, model), stats) in &state.providers {
        let entry = json!({
            "provider": provider,
            "model": model,
            "calls": stats.calls,
            "fails": stats.fails,
            "error_types": stats.error_types,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
        });
        providers
            .entry(kind.clone())
            .or_insert_with(|| serde_json::Value::Array(Vec::new()))
            .as_array_mut()
            .expect("provider bucket is always initialized as an array")
            .push(entry);
    }
    json!({
        "type": "command_summary",
        "user": inner.payload_identity,
        "ts": iso8601_now(),
        "repository": inner.repository,
        "os": python_style_os_name(),
        "chunkhound_version": inner.chunkhound_version,
        "command": state.command,
        "source": state.source,
        "duration_ms": state.started_at.elapsed().as_millis() as u64,
        "success": success,
        "action": state.action,
        "providers": providers,
        "internal_error_type": state.internal_error_type,
    })
}

/// Matches Python's `platform.system()` output (`"Linux"`/`"Darwin"`/
/// `"Windows"`), not Rust's `std::env::consts::OS` (`"linux"`/`"macos"`/
/// `"windows"`) — the wiki's examples and any downstream consumer expect
/// the Python spelling, and this field must be identical regardless of
/// which language recorded a given event.
fn python_style_os_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "Darwin",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    }
}

fn iso8601_now() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

impl Inner {
    fn record_event(&self, event: serde_json::Value) {
        let line = match serde_json::to_string(&event) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("analytics: failed to serialize event, dropping: {e}");
                return;
            }
        };
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let added = line.len() as u64 + 1;
        if buffer.bytes + added > self.config.max_buffer_bytes {
            log::debug!("analytics: buffer size cap reached, dropping event");
            return;
        }
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&buffer.active_path)
            .and_then(|mut f| writeln!(f, "{line}"));
        match result {
            Ok(()) => {
                buffer.lines += 1;
                buffer.bytes += added;
            }
            Err(e) => log::warn!("analytics: failed to write buffer file: {e}"),
        }
    }

    fn maybe_flush(&self, force: bool) {
        let should_flush = {
            let buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            force
                || buffer.lines >= self.config.flush_batch_size
                || (buffer.lines > 0 && buffer.last_flush.elapsed() >= self.config.flush_interval)
        };
        if should_flush {
            self.flush_active();
        }
        self.sweep_orphans();
    }

    /// Atomic rename-before-upload, not read-then-delete: any `record_event`
    /// racing with this rename either lands in the file before the rename
    /// (included in this flush) or starts a fresh active file after it
    /// (included in the next flush) — never lost, never interleaved with the
    /// upload read.
    fn flush_active(&self) {
        let (old_active, rotated_path) = {
            let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let old_active = buffer.active_path.clone();
            let rotated_path = rotated_path_for(&old_active);
            buffer.active_path = new_active_buffer_path(&self.config.buffer_dir);
            buffer.lines = 0;
            buffer.bytes = 0;
            buffer.last_flush = Instant::now();
            (old_active, rotated_path)
        };
        match fs::rename(&old_active, &rotated_path) {
            Ok(()) => self.upload_and_cleanup(&rotated_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Nothing was ever written this cycle; not an error.
            }
            Err(e) => log::warn!("analytics: failed to rotate buffer file: {e}"),
        }
    }

    /// Picks up buffer files left behind by a crashed prior process — never
    /// this recorder's own current active file. Matched by filename pattern
    /// and idle mtime, not PID liveness, since PIDs get reused.
    fn sweep_orphans(&self) {
        let Ok(entries) = fs::read_dir(&self.config.buffer_dir) else {
            return;
        };
        let own_active = self
            .buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_path
            .clone();
        let idle_threshold = self.config.flush_interval * 2;
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if path == own_active {
                continue;
            }
            let is_pending = name.contains(".pending-");
            let is_active_looking =
                name.starts_with("buffer-") && name.ends_with(".jsonl") && !is_pending;
            if !is_pending && !is_active_looking {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            let Ok(age) = SystemTime::now().duration_since(modified) else {
                continue;
            };
            if age < idle_threshold {
                continue;
            }
            if is_pending {
                self.upload_and_cleanup(&path);
            } else {
                let rotated = rotated_path_for(&path);
                match fs::rename(&path, &rotated) {
                    Ok(()) => self.upload_and_cleanup(&rotated),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // Another process's sweep won the race; harmless.
                    }
                    Err(e) => log::warn!("analytics: failed to rotate orphan buffer file: {e}"),
                }
            }
        }
    }

    fn upload_and_cleanup(&self, path: &Path) {
        let Some(target) = &self.config.s3_target else {
            // No S3 configured: leave the rotated file on disk. It's
            // retried on every flush/sweep cycle, which is a harmless no-op
            // until a target is configured.
            return;
        };
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("analytics: failed to read buffer file for upload: {e}");
                return;
            }
        };
        if data.is_empty() {
            let _ = fs::remove_file(path);
            return;
        }
        let key = self.object_key();
        match target.put_object(&self.http, &key, data) {
            Ok(()) => {
                if let Err(e) = fs::remove_file(path) {
                    log::warn!("analytics: uploaded but failed to remove local buffer: {e}");
                }
            }
            Err(e) => log::warn!("analytics: upload failed, will retry next cycle: {e}"),
        }
    }

    fn object_key(&self) -> String {
        use time::format_description::well_known::Rfc3339;
        let now = time::OffsetDateTime::now_utc();
        const DATE_FORMAT: &[time::format_description::BorrowedFormatItem<'_>] =
            time::macros::format_description!("[year]/[month]/[day]");
        let date = now
            .format(DATE_FORMAT)
            .unwrap_or_else(|_| "1970/01/01".to_string());
        let ts = now
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
        let mut suffix = [0u8; 8];
        rand::thread_rng().fill_bytes(&mut suffix);
        let suffix_hex: String = suffix.iter().map(|b| format!("{b:02x}")).collect();
        format!(
            "analytics/{}/{}/{}/{}_{}.jsonl",
            self.repository, self.key_segment, date, ts, suffix_hex
        )
    }
}

fn rotated_path_for(active_path: &Path) -> PathBuf {
    let mut suffix = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut suffix);
    let suffix_hex: String = suffix.iter().map(|b| format!("{b:02x}")).collect();
    let mut rotated = active_path.as_os_str().to_owned();
    rotated.push(format!(".pending-{suffix_hex}"));
    PathBuf::from(rotated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use tempfile::tempdir;

    fn raw_config(buffer_dir: &Path) -> RawConfig {
        RawConfig {
            buffer_dir: buffer_dir.to_path_buf(),
            salt_path: buffer_dir.join("salt"),
            repository_dir: buffer_dir.to_path_buf(),
            ..RawConfig::default()
        }
    }

    fn count_buffer_files(dir: &Path) -> usize {
        fs::read_dir(dir).unwrap().count()
    }

    #[test]
    fn disabled_config_yields_no_inner() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.enabled = false;
        assert!(build_inner_from_raw(raw).is_none());
    }

    #[test]
    fn record_event_appends_a_line_and_updates_counters() {
        let dir = tempdir().unwrap();
        let inner = build_inner_from_raw(raw_config(dir.path())).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.record_event(json!({"type": "command_summary", "n": 2}));

        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(buffer.lines, 2);
        let contents = fs::read_to_string(&buffer.active_path).unwrap();
        assert_eq!(contents.lines().count(), 2);
    }

    #[test]
    fn size_cap_drops_events_without_blocking() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.max_buffer_bytes = 10; // smaller than a single serialized event
        let inner = build_inner_from_raw(raw).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(
            buffer.lines, 0,
            "event over the cap must be dropped, not written"
        );
    }

    #[test]
    fn flush_active_rotates_and_uploads_then_clears_local_buffer() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active();

        mock.assert();
        assert_eq!(
            count_buffer_files(dir.path()),
            0, // the rotated file was uploaded and removed; the fresh active
            // path isn't created on disk until the next record_event()
            "the uploaded rotated file must be removed after a successful upload"
        );
    }

    #[test]
    fn flush_leaves_buffer_intact_when_upload_fails() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(500);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active();

        mock.assert();
        let pending: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .collect();
        assert_eq!(
            pending.len(),
            1,
            "a failed upload must leave the rotated file on disk for the next retry"
        );
    }

    #[test]
    fn no_s3_target_configured_leaves_rotated_file_as_a_harmless_noop() {
        let dir = tempdir().unwrap();
        let inner = build_inner_from_raw(raw_config(dir.path())).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active();

        let pending_count = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .count();
        assert_eq!(pending_count, 1);
    }

    #[test]
    fn orphan_sweep_uploads_a_stale_active_looking_file() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        // idle_threshold = 2 * flush_interval; zero makes every existing
        // file "old enough" immediately, avoiding any need to fake mtimes.
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();

        // Simulate a file left behind by a crashed prior process: not
        // inner's own active file, matching the active-file naming pattern.
        let orphan_path = dir.path().join("buffer-999999-123.jsonl");
        fs::write(&orphan_path, "{\"type\":\"command_summary\"}\n").unwrap();

        inner.sweep_orphans();

        mock.assert();
        assert!(
            !orphan_path.exists(),
            "orphan must be rotated away and uploaded"
        );
    }

    #[test]
    fn orphan_sweep_uploads_a_leftover_pending_file() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();

        // Simulates a process that rotated but crashed before uploading.
        let pending_path = dir.path().join("buffer-999999-123.jsonl.pending-deadbeef");
        fs::write(&pending_path, "{\"type\":\"command_summary\"}\n").unwrap();

        inner.sweep_orphans();

        mock.assert();
        assert!(!pending_path.exists());
    }

    #[test]
    fn orphan_sweep_never_touches_its_own_active_file() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        inner.sweep_orphans();

        let buffer = inner.buffer.lock().unwrap();
        assert!(
            buffer.active_path.exists(),
            "the recorder's own live active file must never be swept as an orphan"
        );
    }

    #[test]
    fn concurrent_record_event_never_loses_a_write_across_a_racing_flush() {
        // Direct test of the atomic-rename-before-upload guarantee: every
        // event recorded either lands in the file rename picks up, or in
        // the fresh file created immediately after -- never both, never
        // neither.
        let dir = tempdir().unwrap();
        let inner = StdArc::new(build_inner_from_raw(raw_config(dir.path())).unwrap());

        let writer_inner = StdArc::clone(&inner);
        let writer = std::thread::spawn(move || {
            for i in 0..500 {
                writer_inner.record_event(json!({"type": "command_summary", "n": i}));
            }
        });
        for _ in 0..20 {
            inner.flush_active();
            std::thread::sleep(Duration::from_micros(200));
        }
        writer.join().unwrap();
        inner.flush_active(); // pick up whatever's left in the final active file

        let mut seen = std::collections::HashSet::new();
        for entry in fs::read_dir(dir.path()).unwrap().flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            if !name.starts_with("buffer-") {
                continue;
            }
            for line in fs::read_to_string(&path).unwrap().lines() {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                seen.insert(value["n"].as_u64().unwrap());
            }
        }
        assert_eq!(
            seen.len(),
            500,
            "every recorded event must appear exactly once, across all flushes"
        );
    }
}
