//! SigV4-signed `PutObject` against an S3-compatible bucket (self-hosted
//! MinIO), reusing the crate's existing blocking `reqwest` client. No
//! `ListObject`/`GetObject`/multipart calls are ever made — one presigned
//! PUT per flush, matching the write-only credential model in `AGENTS.md`'s
//! Authorization & Trust Model section.

use rusty_s3::{Bucket, Credentials, S3Action, UrlStyle};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum S3Error {
    #[error("invalid analytics S3 endpoint/bucket config: {0}")]
    Config(String),
    #[error("analytics upload request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("analytics upload rejected by server: HTTP {0}")]
    Rejected(u16),
}

pub(crate) struct S3Target {
    bucket: Bucket,
    credentials: Credentials,
}

impl S3Target {
    pub fn new(
        endpoint_url: &str,
        bucket_name: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Result<Self, S3Error> {
        let endpoint = endpoint_url
            .parse()
            .map_err(|e| S3Error::Config(format!("invalid endpoint URL: {e}")))?;
        // MinIO's default addressing is path-style (endpoint/bucket/key), not
        // virtual-host-style (bucket.endpoint/key) — a self-hosted instance
        // is not expected to have per-bucket DNS/TLS set up.
        let bucket = Bucket::new(
            endpoint,
            UrlStyle::Path,
            bucket_name.to_string(),
            "us-east-1",
        )
        .map_err(|e| S3Error::Config(format!("invalid bucket config: {e}")))?;
        let credentials = Credentials::new(access_key, secret_key);
        Ok(Self {
            bucket,
            credentials,
        })
    }

    pub fn put_object(
        &self,
        client: &reqwest::blocking::Client,
        object_key: &str,
        body: Vec<u8>,
    ) -> Result<(), S3Error> {
        let action = self.bucket.put_object(Some(&self.credentials), object_key);
        // Short-lived presigned URL covering exactly one flush's PUT — never
        // held for the process lifetime, never persisted.
        let url = action.sign(Duration::from_secs(60));
        let response = client
            .put(url)
            .header("content-type", "application/x-ndjson")
            .body(body)
            .send()?;
        if !response.status().is_success() {
            return Err(S3Error::Rejected(response.status().as_u16()));
        }
        Ok(())
    }
}
