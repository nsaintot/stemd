//! The model catalogue: which artefacts exist, and their pinned digests.
//!
//! All of them are served from this project's own release, pinned by digest, so a
//! stale digest fails loudly on the next fresh install. Re-hosted rather than
//! fetched from upstream so that one release under this project's control is the
//! only thing to keep alive; the digests are unchanged by the move.
//!
//! No manifest travels with the weights. The architecture is compiled into
//! `stemd_mlx` and every layer checks the shape of the tensor it pulls, so a file
//! that is not this architecture fails at load naming the tensor that disagreed.

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
use stemd_core::Family;

/// What the window offers, and what each choice costs.
///
/// Named for the trade rather than the checkpoint. See docs/evaluation.md for the
/// measurements.
///
/// Both are demucs v4. The v3 preset was retired because a bidirectional LSTM is
/// the one demucs architecture MLX runs badly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    /// `htdemucs` (demucs v4). The cheap end, and the model that won a listening
    /// test on real material despite scoring lowest of the old three on MUSDB.
    Fast,
    /// `htdemucs_ft`: four checkpoints, each fine-tuned for one source, in a single
    /// artefact, of which two are run.
    ///
    /// The four sources have no reason to sum to the mix, so shipping `bass + other`
    /// as harmonics put all of that error on the part the player rebuilds. Running the
    /// drums and vocals specialists and shipping harmonics as the remainder removes it.
    Balanced,
    /// BS PolarFormer for the vocals, `htdemucs_ft`'s drums specialist for the rest,
    /// chained. Two artefacts and two models, and the only preset that is not one
    /// architecture.
    ///
    /// It beats both others on every one of the three parts a player drives, and it
    /// is slow. See docs/evaluation.md.
    ///
    /// The vocals half was `bs_roformer_viperx` until PolarFormer replaced it. The
    /// two score the same on MUSDB, and PolarFormer leaves far less voice in the
    /// harmonics on electronic material, which no benchmark here caught.
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

    /// One line for the dropdown. Times are for a 6:38 track on an idle M1 Pro.
    pub const fn detail(self) -> &'static str {
        match self {
            Self::Fast => "htdemucs v4 — ~22 s per track",
            Self::Balanced => "htdemucs ft — ~48 s per track",
            Self::Quality => "PolarFormer + htdemucs ft — ~256 s per track",
        }
    }

    /// True when this preset cannot meet the 50 s target the player expects.
    ///
    /// For a preset that cannot meet the budget by design, not one that can be pushed
    /// past it by load. `Balanced` was marked until it stopped running two of its four
    /// models and came in under the target.
    pub const fn over_budget(self) -> bool {
        matches!(self, Self::Quality)
    }

    /// Artefact name, i.e. what `--demucs-model` takes, what identifies the
    /// preset in the settings file, and what the weights file is called
    /// without its extension.
    ///
    /// [`Self::Quality`] needs two artefacts and this is the one that names it;
    /// see [`Self::artefacts`].
    pub const fn artefact(self) -> &'static str {
        match self {
            Self::Fast => "htdemucs",
            Self::Balanced => "htdemucs_ft",
            Self::Quality => "bs_polarformer",
        }
    }

    /// Every artefact the preset needs loaded, the naming one first.
    ///
    /// Only [`Self::Quality`] has more than one, and its second is the same
    /// `htdemucs_ft` [`Self::Balanced`] uses, so a machine with Balanced
    /// installed only downloads the difference.
    pub const fn artefacts(self) -> &'static [&'static str] {
        match self {
            Self::Fast => &["htdemucs"],
            Self::Balanced => &["htdemucs_ft"],
            Self::Quality => &["bs_polarformer", "htdemucs_ft"],
        }
    }

    pub const fn source(self) -> &'static ModelSource {
        match self {
            Self::Fast => &FAST_SOURCE,
            Self::Balanced => &BALANCED_SOURCE,
            Self::Quality => &QUALITY_SOURCE,
        }
    }

    /// How the preset uses its weights, bumped when that changes.
    ///
    /// The weights identify themselves by digest, which is enough while a
    /// preset is only ever "run this file". It stopped being enough when
    /// `Balanced` went from running all four of its models and shipping
    /// `bass + other` to running two and shipping the remainder: different
    /// audio out of the same bytes in. Cached stems are keyed on the model's
    /// identity, so without this they would survive a change that invalidates
    /// them.
    /// The architectures this preset runs, in the order it runs them.
    ///
    /// Not a detail of the weights: which family a half is decides what
    /// precision suits it, and the two answer differently on CUDA. `None` is a
    /// hand-named artefact, which is always a single demucs.
    pub const fn families(preset: Option<Self>) -> &'static [Family] {
        match preset {
            Some(Self::Quality) => &[Family::Roformer, Family::HtDemucs],
            Some(Self::Balanced) => &[Family::HtDemucs, Family::HtDemucs],
            Some(Self::Fast) | None => &[Family::HtDemucs],
        }
    }

    pub const fn recipe(self) -> u32 {
        match self {
            Self::Fast => 1,
            // 1 was all four models with harmonics as `bass + other`.
            Self::Balanced => 2,
            // 1 ran both halves over the mixture. 2 chains them: the drums
            // half is handed `mix - vocals`. Same weights, same two forward
            // passes, different audio out.
            Self::Quality => 2,
        }
    }

    /// What identifies this preset's weights, for a cache key.
    ///
    /// The pinned digest of the weights, so repointing a preset at a different
    /// artefact invalidates its cached separations by itself. Not the whole
    /// story on its own: see [`Self::recipe`].
    pub fn digest(self) -> &'static str {
        self.source()
            .files
            .iter()
            .find(|f| f.name.ends_with(WEIGHTS_EXTENSION))
            .map_or_else(|| self.artefact(), |f| f.sha256)
    }

    /// Match an artefact name back to a preset, so `--demucs-model` and the
    /// window cannot disagree about what is loaded.
    pub fn from_artefact(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.artefact() == name)
    }

    pub fn total_bytes(self) -> u64 {
        self.source().files.iter().map(|f| f.bytes).sum()
    }
}

/// The preset loaded when `--demucs-model` is not given.
pub const DEFAULT_PRESET: Preset = Preset::Fast;

/// What a weights file is called. One place, because both the catalogue and the
/// loader have to agree on it.
pub const WEIGHTS_EXTENSION: &str = ".safetensors";

/// Retired artefacts, still named so a settings file that mentions one can say
/// what happened rather than falling back in silence.
///
/// Three are TorchScript and none of them loads: there is no TorchScript runtime
/// left to load them with.
///
/// `bs_roformer_viperx` is retired for a different reason, and the distinction is
/// worth keeping: nothing is wrong with it, it stopped being the best answer. It
/// no longer loads either, because a lone vocals model is not a separator, and
/// only a preset can say which drums half to pair it with.
pub const RETIRED: [&str; 4] = [
    "hdemucs_mmi_mps",
    "htdemucs_mps",
    "htdemucs_ft_mps",
    "bs_roformer_viperx",
];

pub const FAST_SOURCE: ModelSource = ModelSource {
    files: &[RemoteFile {
        name: "htdemucs.safetensors",
        url: "https://github.com/nsaintot/stemd/releases/download/models-v2/htdemucs.safetensors",
        sha256: "339d267a7a6983a11eedbdc00413c602a65e9b9103f695fb5c2b2a481cd9d297",
        bytes: 168_005_865,
    }],
};

pub const BALANCED_SOURCE: ModelSource = ModelSource {
    files: &[BALANCED_WEIGHTS],
};

const BALANCED_WEIGHTS: RemoteFile = RemoteFile {
    name: "htdemucs_ft.safetensors",
    url: "https://github.com/nsaintot/stemd/releases/download/models-v2/htdemucs_ft.safetensors",
    sha256: "53f03b1ad4b4d211025a35da65460ba61a17547adf9c0544cad0ebcc8d7bbabb",
    bytes: 672_024_519,
};

/// BS PolarFormer, plus the same `htdemucs_ft` Balanced uses.
///
/// Uploaded and verified from the public URL: the bytes the network serves hash to
/// the digest below, which is the one thing a pin cannot check about itself.
/// Converted by `tools/export/convert_roformer.py` from ZFTurbo's
/// `model_bs_polarformer_float16.ckpt`, the only form its weights are published in.
pub const QUALITY_SOURCE: ModelSource = ModelSource {
    files: &[
        RemoteFile {
            name: "bs_polarformer.safetensors",
            url: "https://github.com/nsaintot/stemd/releases/download/models-v2/bs_polarformer.safetensors",
            sha256: "9e08a5e075204e893a4eb393ae64d47177c76f11a306686db1343d7cc7c468f6",
            bytes: 102_201_832,
        },
        BALANCED_WEIGHTS,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_and_artefacts_round_trip() {
        for preset in Preset::ALL {
            assert_eq!(Preset::from_artefact(preset.artefact()), Some(preset));
            // Every file of a preset must belong to one of its artefacts,
            // which is what keeps the download and the loader in step.
            assert!(
                preset.source().files.iter().all(|f| preset
                    .artefacts()
                    .iter()
                    .any(|name| f.name.starts_with(name))),
                "{preset:?} has files matching none of {:?}",
                preset.artefacts()
            );
        }
        assert_eq!(Preset::from_artefact("nonsense"), None);
    }

    /// One weights file per artefact the preset names, in the same order.
    ///
    /// This is what lets the loader look for `<artefact>.safetensors` without
    /// consulting the catalogue, and what makes the digest unambiguous.
    #[test]
    fn every_artefact_has_its_weights_file() {
        for preset in Preset::ALL {
            let files = preset.source().files;
            let names = preset.artefacts();
            assert_eq!(files.len(), names.len(), "{preset:?} is inconsistent");
            for (file, name) in files.iter().zip(names) {
                assert_eq!(file.name, format!("{name}{WEIGHTS_EXTENSION}"));
            }
            assert_eq!(preset.artefact(), names[0], "{preset:?} names its first");
            assert_eq!(preset.digest(), files[0].sha256);
        }
    }

    /// An artefact used by more than one preset must be pinned identically by
    /// all of them, or one would fetch a file the next rejects as mismatched.
    ///
    /// Three presets share `htdemucs_ft` now. Written over every file every
    /// preset names rather than over a pair, so a fifth preset is covered by
    /// existing code instead of needing a line added here.
    #[test]
    fn a_shared_artefact_is_pinned_once() {
        let mut seen: std::collections::HashMap<&str, (&str, &str, u64)> =
            std::collections::HashMap::new();
        for preset in Preset::ALL {
            for file in preset.source().files {
                let pin = (file.sha256, file.url, file.bytes);
                if let Some(first) = seen.insert(file.name, pin) {
                    assert_eq!(
                        first, pin,
                        "{} is pinned differently by {preset:?} than by an \
                         earlier preset",
                        file.name
                    );
                }
            }
        }
        assert!(
            seen.len() < Preset::ALL.iter().map(|p| p.source().files.len()).sum(),
            "no artefact is shared, so this test checked nothing"
        );
    }

    /// A retired artefact must not quietly match a preset again, or a settings
    /// file naming it would load something other than what it says.
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

    /// Every demucs artefact any preset names is v4, which is what lets one runtime
    /// serve them all. A v3 artefact would put the LSTM back.
    ///
    /// Stated over artefacts rather than presets, and by what the name says rather
    /// than by a list of presets to skip: a skip list names exceptions that go stale
    /// as presets are added.
    #[test]
    fn every_demucs_artefact_is_v4() {
        for preset in Preset::ALL {
            for artefact in preset.artefacts() {
                // v3 is `hdemucs_*`, v4 is `htdemucs*`. Anything else is not a
                // demucs at all and this test has no opinion on it.
                assert!(
                    !artefact.starts_with("hdemucs"),
                    "{preset:?} names {artefact}, which is demucs v3"
                );
            }
        }
        // And the guard is not vacuous: the name it exists to reject would be.
        assert!(RETIRED.iter().any(|name| name.starts_with("hdemucs")));
    }

    /// Every artefact comes from one release over https, at a URL ending in the
    /// filename the loader will look for.
    ///
    /// That last is not pedantry: the loader finds a model by looking for
    /// `<artefact>.safetensors` on disk and the downloader saves under `file.name`, so
    /// a URL whose last segment disagreed would download something the loader cannot
    /// see.
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
            // A placeholder digest here would mean shipping a build that
            // accepts whatever the network returns.
            assert_eq!(file.sha256.len(), 64, "{} needs a real sha256", file.name);
            assert!(
                file.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{} sha256 must be hex",
                file.name
            );
        }
    }
}
