//! The terms of each model, written beside its weights after a download.
//!
//! The texts live in `resources/model-terms`, where `scripts/bundle-app.sh` also reads them.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::preset::WEIGHTS_EXTENSION;

/// The suffix of a terms file: `htdemucs.LICENSE.txt` sits beside `htdemucs.safetensors`.
pub const TERMS_SUFFIX: &str = ".LICENSE.txt";

const STEMD_ROFORMER: &str = include_str!("../../../../resources/model-terms/stemd_roformer.txt");
const DEMUCS: &str = include_str!("../../../../resources/model-terms/demucs.txt");
const ROFORMER_CONVERSION: &str =
    include_str!("../../../../resources/model-terms/roformer-conversion.txt");

/// The terms of an artefact, such as `htdemucs`.
pub fn of(artefact: &str) -> Option<&'static str> {
    match artefact {
        "stemd_roformer" => Some(STEMD_ROFORMER),
        "htdemucs" | "htdemucs_ft" => Some(DEMUCS),
        "bs_polarformer" | "bs_roformer_viperx" => Some(ROFORMER_CONVERSION),
        _ => None,
    }
}

/// The terms file for the weights file `weights` in `dir`, or `None` for a file with no known
/// terms.
fn beside(dir: &Path, weights: &str) -> Option<(PathBuf, &'static str)> {
    let artefact = weights.strip_suffix(WEIGHTS_EXTENSION)?;
    let text = of(artefact)?;
    Some((dir.join(format!("{artefact}{TERMS_SUFFIX}")), text))
}

/// Write the terms of the weights file `weights` beside it in `dir`.
pub fn write(dir: &Path, weights: &str) -> Result<()> {
    let Some((path, text)) = beside(dir, weights) else {
        return Ok(());
    };
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))
}

/// [`write()`] when the terms file is missing, for weights downloaded by an older build.
pub fn ensure(dir: &Path, weights: &str) -> Result<()> {
    match beside(dir, weights) {
        Some((path, _)) if !path.is_file() => write(dir, weights),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::preset::{FORMER, Preset, RETIRED};
    use super::*;

    #[test]
    fn every_downloaded_artefact_has_terms() {
        for preset in Preset::ALL {
            assert!(of(preset.artefact()).is_some(), "{preset:?} has no terms");
        }
        assert!(of(FORMER[0].0).is_some());
        assert!(
            RETIRED
                .iter()
                .any(|name| of(name) == Some(ROFORMER_CONVERSION))
        );
        assert!(
            STEMD_ROFORMER.contains("https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode")
        );
        assert!(ROFORMER_CONVERSION.contains("Its authors publish no terms."));
    }

    #[test]
    fn the_terms_sit_beside_the_weights_and_integrity_ignores_them() {
        let dir = crate::scratch::Dir::new("terms");
        let preset = Preset::Quality;
        let weights = preset.source().files[0].name;
        write(&dir, weights).unwrap();
        let written = std::fs::read_to_string(dir.join("stemd_roformer.LICENSE.txt")).unwrap();
        assert_eq!(written, STEMD_ROFORMER);

        // The terms file is not the artefact, and hashing reports only the weights.
        assert!(!super::super::is_complete(&dir, preset.source()));
        assert_eq!(super::super::integrity::mismatched(&dir, preset), [weights]);

        // An existing terms file is left alone, and a file with no terms writes nothing.
        std::fs::write(dir.join("stemd_roformer.LICENSE.txt"), "kept").unwrap();
        ensure(&dir, weights).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("stemd_roformer.LICENSE.txt")).unwrap(),
            "kept"
        );
        write(&dir, "unknown.safetensors").unwrap();
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
