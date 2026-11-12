//! The newest published app release, from GitHub's API.
// macOS builds this for its tests only.
#![cfg_attr(target_os = "macos", allow(dead_code))]

use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

/// The most recent releases. Model releases such as `models-v3` are in this list
/// and can be GitHub's "latest", so [`latest`] picks the newest `v*` tag.
const RELEASES: &str = "https://api.github.com/repos/nsaintot/stemd/releases?per_page=30";

/// A release, as the API describes it.
#[derive(Clone, Debug, Deserialize)]
pub struct Release {
    pub tag_name: String,
    pub html_url: String,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub prerelease: bool,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

/// One file of a release.
#[derive(Clone, Debug, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
    pub size: u64,
    /// `sha256:<hex>`, which GitHub computes on upload.
    #[serde(default)]
    pub digest: Option<String>,
}

impl Release {
    /// The version, without the tag's `v`.
    pub fn version(&self) -> &str {
        self.tag_name.trim_start_matches('v')
    }

    /// The Windows installer, `stemd-<version>-setup.exe`.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn installer(&self) -> Option<&Asset> {
        let name = format!("stemd-{}-setup.exe", self.version());
        self.assets.iter().find(|a| a.name == name)
    }
}

impl Asset {
    /// The SHA-256 GitHub recorded, as lowercase hex.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub fn sha256(&self) -> Option<String> {
        let hex = self.digest.as_deref()?.strip_prefix("sha256:")?;
        (hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| hex.to_ascii_lowercase())
    }
}

/// Fetch the newest published app release: a `v<version>` tag, neither a
/// draft nor a pre-release.
pub fn latest() -> Result<Release> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .build()
        .into();
    let body = agent
        .get(RELEASES)
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", concat!("stemd/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|e| anyhow::anyhow!("GET {RELEASES}: {e}"))?
        .into_body()
        .read_to_string()
        .context("reading the releases")?;
    let releases: Vec<Release> = serde_json::from_str(&body).context("parsing the releases")?;
    newest(releases).context("no published app release")
}

/// The newest of `releases` that is an app release.
fn newest(releases: Vec<Release>) -> Option<Release> {
    releases
        .into_iter()
        .filter(|r| !r.draft && !r.prerelease && r.tag_name.starts_with('v'))
        .filter_map(|r| Some((super::Version::parse(&r.tag_name)?, r)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, r)| r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
        "tag_name": "v0.2.0",
        "html_url": "https://github.com/nsaintot/stemd/releases/tag/v0.2.0",
        "draft": false,
        "assets": [
            {"name": "stemd-0.2.0-windows-x64.zip", "size": 1, "digest": null,
             "browser_download_url": "https://example.invalid/zip"},
            {"name": "stemd-0.2.0-setup.exe", "size": 84000000,
             "digest": "sha256:ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
             "browser_download_url": "https://example.invalid/setup"}
        ]
    }"#;

    #[test]
    fn finds_the_installer_and_its_digest() {
        let release: Release = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(release.version(), "0.2.0");
        let setup = release.installer().unwrap();
        assert_eq!(setup.browser_download_url, "https://example.invalid/setup");
        assert_eq!(
            setup.sha256().unwrap(),
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        );
    }

    #[test]
    fn picks_the_newest_app_release() {
        let releases: Vec<Release> = serde_json::from_str(
            r#"[
                {"tag_name": "models-v4", "html_url": "m"},
                {"tag_name": "v0.3.0", "html_url": "draft", "draft": true},
                {"tag_name": "v0.2.1-rc1", "html_url": "pre", "prerelease": true},
                {"tag_name": "v0.1.9", "html_url": "old"},
                {"tag_name": "v0.2.0", "html_url": "new"}
            ]"#,
        )
        .unwrap();
        assert_eq!(newest(releases).unwrap().html_url, "new");
        assert!(newest(Vec::new()).is_none());
    }

    #[test]
    fn a_missing_or_malformed_digest_is_none() {
        let mut asset: Asset =
            serde_json::from_str(r#"{"name": "a", "size": 1, "browser_download_url": "u"}"#)
                .unwrap();
        assert_eq!(asset.sha256(), None);
        asset.digest = Some("sha512:00".into());
        assert_eq!(asset.sha256(), None);
        asset.digest = Some("sha256:00".into());
        assert_eq!(asset.sha256(), None);
    }
}
