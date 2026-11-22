//! The model catalogue: which artefacts exist, and their pinned digests.
//!
//! Every artefact is served from this project's own releases and pinned by digest, so a stale
//! digest fails on the next fresh install. The project hosts the files so that its own releases are
//! the only thing to keep available; the digests are the same as upstream.
//!
//! No manifest comes with the weights. The architecture is compiled into `stemd_mlx`, and every
//! layer checks the shape of each tensor it reads, so a file with another architecture fails at
//! load with the name of the tensor that disagreed.

use stemd_core::Family;

/// One file of a model artefact, pinned by digest.
pub struct RemoteFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
}

/// Everything needed to reconstruct a model artefact from the network. The
/// artefact name lives on [`Preset`], so the two cannot drift.
pub struct ModelSource {
    pub files: &'static [RemoteFile],
}

/// The presets the window offers. Each is named for its speed and quality trade-off;
/// docs/evaluation.md has the measurements.
///
/// Fast and Balanced are demucs v4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// `htdemucs` (demucs v4), the cheapest preset.
    Fast,
    /// `htdemucs_ft`: one artefact with four checkpoints, each fine-tuned for one source; two of
    /// them run.
    ///
    /// The four sources need not sum to the mix, so shipping `bass + other` as harmonics would put
    /// that error on the part the player rebuilds. Running the drums and vocals specialists and
    /// shipping harmonics as the remainder removes the error.
    Balanced,
    /// stemd's own 4-stem BS-RoFormer. It ships drums and vocals, and harmonics is the remainder.
    /// See docs/evaluation.md.
    Quality,
}

impl Preset {
    pub const ALL: [Self; 3] = [Self::Fast, Self::Balanced, Self::Quality];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::Quality => "Quality",
        }
    }

    /// One line about the model. Times are for a 6:38 track on an idle M1 Pro.
    pub const fn detail(self) -> &'static str {
        match self {
            Self::Fast => "htdemucs v4 · about 22 s per track",
            Self::Balanced => "htdemucs ft · about 48 s per track",
            Self::Quality => "Stemd RoFormer · about 110 s per track",
        }
    }

    /// The artefact name: the value `--demucs-model` takes, the preset's name in the settings file,
    /// and the weights file's name without its extension.
    pub const fn artefact(self) -> &'static str {
        match self {
            Self::Fast => "htdemucs",
            Self::Balanced => "htdemucs_ft",
            Self::Quality => "stemd_roformer",
        }
    }

    pub const fn source(self) -> &'static ModelSource {
        match self {
            Self::Fast => &FAST_SOURCE,
            Self::Balanced => &BALANCED_SOURCE,
            Self::Quality => &QUALITY_SOURCE,
        }
    }

    /// The architectures this preset runs, in the order it runs them.
    ///
    /// The family of each model decides which precision suits it, and the families differ on CUDA.
    /// `None` stands for a hand-named artefact, which is always a single demucs.
    pub const fn families(preset: Option<Self>) -> &'static [Family] {
        match preset {
            Some(Self::Quality) => &[Family::Roformer],
            Some(Self::Balanced) => &[Family::HtDemucs, Family::HtDemucs],
            Some(Self::Fast) | None => &[Family::HtDemucs],
        }
    }

    /// A version of how the preset uses its weights, part of the cache key. Bump it when that use
    /// changes, since the weights digest stays the same.
    pub const fn recipe(self) -> u32 {
        match self {
            Self::Fast => 1,
            Self::Balanced => 2,
            Self::Quality => 3,
        }
    }

    /// The pinned digest of this preset's weights, for a cache key.
    ///
    /// Pointing a preset at a different artefact changes the digest, which invalidates its cached
    /// separations. See also [`Self::recipe`].
    pub fn digest(self) -> &'static str {
        self.source()
            .files
            .iter()
            .find(|f| f.name.ends_with(WEIGHTS_EXTENSION))
            .map_or_else(|| self.artefact(), |f| f.sha256)
    }

    /// Match an artefact name to a preset, so `--demucs-model` and the window agree on what is
    /// loaded. A preset's former name also matches it, so a settings file saved before a rename
    /// keeps its choice.
    pub fn from_artefact(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.artefact() == name)
            .or_else(|| {
                FORMER
                    .iter()
                    .find(|(former, _)| *former == name)
                    .map(|(_, preset)| *preset)
            })
    }

    pub fn total_bytes(self) -> u64 {
        self.source().files.iter().map(|f| f.bytes).sum()
    }
}

/// The preset loaded when `--demucs-model` is not given.
pub const DEFAULT_PRESET: Preset = Preset::Fast;

/// The weights file extension, defined once because the catalogue and the loader must agree on it.
pub const WEIGHTS_EXTENSION: &str = ".safetensors";

/// Artefacts a preset used to name, and the preset that replaced them.
const FORMER: [(&str, Preset); 1] = [("bs_polarformer", Preset::Quality)];

/// Retired artefacts, still listed so that a settings file naming one gets an explanation when it
/// falls back.
///
/// Three are TorchScript, which nothing loads. `bs_roformer_viperx` is a vocals-only model and
/// cannot separate on its own.
pub const RETIRED: [&str; 4] = [
    "hdemucs_mmi_mps",
    "htdemucs_mps",
    "htdemucs_ft_mps",
    "bs_roformer_viperx",
];

const FAST_SOURCE: ModelSource = ModelSource {
    files: &[RemoteFile {
        name: "htdemucs.safetensors",
        url: "https://github.com/nsaintot/stemd/releases/download/models-v2/htdemucs.safetensors",
        sha256: "339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297",
        bytes: 168_005_865,
    }],
};

const BALANCED_SOURCE: ModelSource = ModelSource {
    files: &[BALANCED_WEIGHTS],
};

const BALANCED_WEIGHTS: RemoteFile = RemoteFile {
    name: "htdemucs_ft.safetensors",
    url: "https://github.com/nsaintot/stemd/releases/download/models-v2/htdemucs_ft.safetensors",
    sha256: "53f03b1ad4b4d211025a35da65460ba61a17547adf9c0544cad0ebcc8d7bbabb",
    bytes: 672_024_519,
};

/// stemd's own 4-stem BS-RoFormer, float16.
///
/// Converted by `tools/export/convert_stemd_roformer.py` and then
/// `tools/export/convert_roformer.py`.
const QUALITY_SOURCE: ModelSource = ModelSource {
    files: &[RemoteFile {
        name: "stemd_roformer.safetensors",
        url: "https://github.com/nsaintot/stemd/releases/download/models-v3/stemd_roformer.safetensors",
        sha256: "ce3802b634d04d955626cffe038ec5ac8f26ef1c90f3d4cfe5ac542ad6da4b37",
        bytes: 250_585_752,
    }],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_artefacts_round_trip() {
        for preset in Preset::ALL {
            assert_eq!(Preset::from_artefact(preset.artefact()), Some(preset));
            // Every file of a preset belongs to its artefact, which keeps the download and the
            // loader in step.
            assert!(
                preset
                    .source()
                    .files
                    .iter()
                    .all(|f| f.name.starts_with(preset.artefact())),
                "{preset:?} has files not named for {}",
                preset.artefact()
            );
        }
        assert_eq!(Preset::from_artefact("nonsense"), None);
    }

    /// One weights file per preset, named for its artefact.
    ///
    /// This lets the loader look for `<artefact>.safetensors` without the catalogue, and makes the
    /// digest unambiguous.
    #[test]
    fn every_artefact_has_its_weights_file() {
        for preset in Preset::ALL {
            let files = preset.source().files;
            assert_eq!(files.len(), 1, "{preset:?} is inconsistent");
            assert_eq!(
                files[0].name,
                format!("{}{WEIGHTS_EXTENSION}", preset.artefact())
            );
            assert_eq!(preset.digest(), files[0].sha256);
        }
    }

    /// A settings file saved before Quality changed models still opens on Quality,
    /// and the former name is not also listed as retired.
    #[test]
    fn a_former_artefact_still_names_its_preset() {
        for (name, preset) in FORMER {
            assert_eq!(Preset::from_artefact(name), Some(preset));
            assert!(
                !RETIRED.contains(&name),
                "{name} is both former and retired"
            );
        }
    }

    /// A retired artefact never matches a preset, so a settings file naming it never loads a
    /// different model.
    #[test]
    fn a_retired_artefact_is_not_a_preset() {
        for name in RETIRED {
            assert_eq!(
                Preset::from_artefact(name),
                None,
                "{name} is still a preset"
            );
        }
    }

    /// Every demucs artefact that a preset names is v4, so one runtime serves them all; a v3
    /// artefact would bring back the LSTM.
    ///
    /// The test checks artefact names, so it needs no list of presets to skip, which would go stale
    /// as presets are added.
    #[test]
    fn every_demucs_artefact_is_v4() {
        for preset in Preset::ALL {
            // v3 is `hdemucs_*` and v4 is `htdemucs*`. Other names are not demucs, and this test
            // ignores them.
            assert!(
                !preset.artefact().starts_with("hdemucs"),
                "{preset:?} names {}, which is demucs v3",
                preset.artefact()
            );
        }
        // A retired name fails this check.
        assert!(RETIRED.iter().any(|name| name.starts_with("hdemucs")));
    }

    /// Every artefact comes from one release over https, at a URL ending in the
    /// filename the loader will look for.
    ///
    /// The loader finds a model by looking for `<artefact>.safetensors` on disk and the downloader
    /// saves under `file.name`, so a URL whose last segment differed would download a file the
    /// loader cannot find.
    #[test]
    fn every_file_is_pinned_over_https() {
        for file in Preset::ALL.iter().flat_map(|p| p.source().files) {
            assert!(
                file.url.starts_with("https://"),
                "{} must be https",
                file.name
            );
            assert!(
                file.url.ends_with(file.name),
                "{} url must end in its filename",
                file.name
            );
            // A placeholder digest would ship a build that accepts whatever the network returns.
            assert_eq!(file.sha256.len(), 64, "{} needs a real sha256", file.name);
            assert!(
                file.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} sha256 must be hex",
                file.name
            );
        }
    }
}
