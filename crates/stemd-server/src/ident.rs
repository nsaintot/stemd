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
//! An id outside that charset, or longer than the buffer, becomes a different
//! string on the client than the one the server reported, silently, and in a way
//! that survives into the client's cache layout. So the server emits only ids that
//! pass through unchanged.

use sha2::{Digest, Sha256};

/// Longest id the server will emit, excluding the NUL a C client terminates with.
///
/// 31 rather than 32 so the whole thing, id plus terminator, fits a 32-byte
/// buffer. A client that truncates reintroduces exactly the divergence this module
/// exists to prevent.
pub const MAX_ID: usize = 31;

/// Hex characters of a digest used to distinguish otherwise equal names.
///
/// 32 bits is a wide margin for the handful of artefacts that exist; this is
/// the trade a short commit hash makes.
pub const DIGEST_CHARS: usize = 8;

/// True for characters a client keeps verbatim.
pub const fn is_portable(c: char) -> bool {
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

/// `<name>-<digest>`, with `name` sanitised and shortened so the whole thing
/// fits [`MAX_ID`].
///
/// The digest covers the *original* name, not the shortened one, so shortening
/// cannot collapse two distinct artefacts onto one id.
pub fn tagged(prefix: &str, name: &str) -> String {
    // prefix, a separator, the name, a separator, the digest.
    let overhead = prefix.len() + 2 + DIGEST_CHARS;
    let room = MAX_ID.saturating_sub(overhead);
    let head: String = portable(name).chars().take(room).collect();
    format!("{prefix}-{head}-{}", short_digest(name))
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
    fn sanitising_replaces_rather_than_drops() {
        // Dropping would shorten the id and could collide two names; the client
        // substitutes, so the server has to substitute identically.
        assert_eq!(portable("my model/v2"), "my_model_v2");
        assert_eq!(portable("plain-1.0_x"), "plain-1.0_x");
        // `.` is in the charset, so a separator becomes `_` but the dots stay.
        assert_eq!(portable("../etc"), ".._etc");
    }

    /// The client turns these into directory names, and `.` is a legal
    /// character, so `..` survives sanitising. The prefix is what guarantees a
    /// segment is never exactly `.` or `..`.
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

    /// `--demucs-model` reaches [`tagged`] unfiltered, and the client no longer
    /// sanitises what it receives, this test is the only thing standing between
    /// arbitrary input and a directory name, so it is deliberately hostile.
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
            // Output is pure ASCII, so a C client's byte count matches this
            // one: `MAX_ID` would not bound anything otherwise.
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
