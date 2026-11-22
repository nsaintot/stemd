//! The identifier contract shared with clients.
//!
//! Ids the server hands out become directory names and config entries on the
//! client, which copies them into a fixed buffer and rewrites anything outside a
//! conservative charset:
//!
//! ```c
//! if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
//!     (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_')
//!     g_stem_sep_id[k] = c;
//! else
//!     g_stem_sep_id[k] = '_';
//! ```
//!
//! An id outside that charset, or longer than the buffer, silently becomes a different string on
//! the client, and the changed string ends up in the client's cache layout. The server therefore
//! emits only ids that pass through unchanged.

use sha2::{Digest, Sha256};
use stemd_core::Audio;

/// Behaviors added after protocol 2, listed in `/v2/health` and the TXT record.
pub const FEATURES: &[&str] = &[];

/// The longest id the server emits, excluding the NUL that a C client appends.
///
/// 31 lets the id and its terminator fit a 32-byte buffer; a client that truncated an id would
/// change it.
const MAX_ID: usize = 31;

/// The number of hex characters of a digest used to tell otherwise equal names apart.
///
/// 32 bits is a wide margin for the handful of artefacts that exist, as with a short commit hash.
pub const DIGEST_CHARS: usize = 8;

/// True for characters a client keeps verbatim.
const fn is_portable(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')
}

/// True when every character of `id` survives a client round trip.
pub fn is_portable_id(id: &str) -> bool {
    id.chars().all(is_portable) && id.len() <= MAX_ID
}

/// Rewrite `name` into the portable charset, exactly as a client would.
pub fn portable(name: &str) -> String {
    name.chars()
        .map(|c| if is_portable(c) { c } else { '_' })
        .collect()
}

/// The leading [`DIGEST_CHARS`] of the SHA-256 of `text`.
///
/// Used to pin a name that had to be shortened, so two names sharing a prefix
/// cannot end up sharing an id.
pub fn short_digest(text: &str) -> String {
    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    digest[..DIGEST_CHARS].to_owned()
}

/// `<prefix>-<name>-<digest>`, with `name` sanitised and shortened so the id fits [`MAX_ID`].
///
/// The digest covers the original name, so shortening cannot give two distinct artefacts one id.
pub fn tagged(prefix: &str, name: &str) -> String {
    // The prefix, a separator, the name, a separator and the digest.
    let overhead = prefix.len() + 2 + DIGEST_CHARS;
    let room = MAX_ID.saturating_sub(overhead);
    let head: String = portable(name).chars().take(room).collect();
    format!("{prefix}-{head}-{}", short_digest(name))
}

/// The number of hex characters in a track id.
pub const TRACK_ID_LEN: usize = 24;

/// Changes when the track id's definition does, so old ids stop matching.
const TRACK_ID_EPOCH: &str = "stemd-track-1";

/// A track's identity: SHA-256 over its samples at the model's rate.
///
/// The hash covers the length-prefixed fields `stemd-track-1`, the rate and the channel count, then
/// every frame's samples as little-endian f32, channel by channel within a frame. The id is the
/// first [`TRACK_ID_LEN`] hex characters.
pub fn track_id(audio: &Audio) -> String {
    let mut hasher = Sha256::new();
    for field in [
        TRACK_ID_EPOCH.to_owned(),
        audio.sample_rate.to_string(),
        audio.channels().to_string(),
    ] {
        hasher.update(u32::try_from(field.len()).unwrap_or(u32::MAX).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    let mut buffer = Vec::with_capacity(1 << 16);
    for i in 0..audio.frames() {
        for channel in &audio.data {
            buffer.extend_from_slice(&channel[i].to_le_bytes());
        }
        if buffer.len() >= 1 << 16 {
            hasher.update(&buffer);
            buffer.clear();
        }
    }
    hasher.update(&buffer);
    let mut id = format!("{:x}", hasher.finalize());
    id.truncate(TRACK_ID_LEN);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_charset_matches_the_clients() {
        for c in ['a', 'Z', '0', '9', '-', '.', '_'] {
            assert!(is_portable(c), "{c:?} should survive a round trip");
        }
        for c in ['/', ' ', ':', '\\', 'é', '\n', '*'] {
            assert!(!is_portable(c), "{c:?} would be rewritten by the client");
        }
    }

    #[test]
    fn sanitising_replaces_invalid_characters_without_dropping_any() {
        // Dropping characters would shorten the id and could make two names collide; the client
        // substitutes them, so the server substitutes them the same way.
        assert_eq!(portable("my model/v2"), "my_model_v2");
        assert_eq!(portable("plain-1.0_x"), "plain-1.0_x");
        // `.` is in the charset, so a separator becomes `_` but the dots stay.
        assert_eq!(portable("../etc"), ".._etc");
    }

    /// The client turns these into directory names, and `.` is a legal character, so `..` survives
    /// sanitising. The prefix guarantees that a segment is never exactly `.` or `..`.
    #[test]
    fn a_tagged_id_is_never_a_path_traversal_component() {
        for name in ["..", ".", "../..", "../../etc/passwd"] {
            let id = tagged("custom", name);
            assert!(is_portable_id(&id), "{id}");
            assert!(
                id != ".." && id != "." && !id.contains('/'),
                "{name:?} produced {id:?}"
            );
            assert!(id.starts_with("custom-"));
        }
    }

    #[test]
    fn a_tagged_id_is_portable_and_bounded() {
        let id = tagged(
            "custom",
            "an extremely long artefact name/with slashes and spaces",
        );
        assert!(is_portable_id(&id), "{id}");
        assert!(id.len() <= MAX_ID, "{} chars: {id}", id.len());
        assert!(id.starts_with("custom-"));
    }

    /// `--demucs-model` reaches [`tagged`] unfiltered and the client does not sanitise what it
    /// receives, so this test feeds hostile input and checks that every id is a safe directory
    /// name.
    #[test]
    fn no_input_can_produce_an_unportable_id() {
        let mut names: Vec<String> = vec![
            String::new(),
            " ".into(),
            "/".repeat(200),
            "\0embedded nul".into(),
            "\n\r\t".into(),
            "../../../etc/passwd".into(),
            ".".repeat(64),
            "üñïçø∂é".into(),
            "🎛️🎚️".into(),
            "with\"quotes'and`ticks".into(),
            "semi;colon&amp|pipe".into(),
            "%2e%2e%2f".into(),
            "a".repeat(10_000),
        ];
        // Every byte value on its own, so no single character can slip through.
        names.extend((0u8..=255).map(|b| String::from_utf8_lossy(&[b]).into_owned()));

        for name in &names {
            let id = tagged("custom", name);
            assert!(
                is_portable_id(&id),
                "{name:?} produced {id:?} ({} chars)",
                id.len()
            );
            // The output is pure ASCII, so a C client's byte count matches this one and `MAX_ID`
            // bounds it.
            assert!(id.is_ascii(), "{id:?} is not ascii");
            assert_eq!(id.len(), id.chars().count());
        }
    }

    #[test]
    fn shortening_cannot_collapse_two_artefacts_onto_one_id() {
        // Both truncate to the same head, so only the digest keeps them apart.
        let a = tagged("custom", "a_very_long_shared_prefix_variant_one");
        let b = tagged("custom", "a_very_long_shared_prefix_variant_two");
        assert_ne!(a, b, "two artefacts must not share a cache identity");
    }

    #[test]
    fn the_same_name_always_produces_the_same_id() {
        assert_eq!(tagged("custom", "steady"), tagged("custom", "steady"));
    }
}
