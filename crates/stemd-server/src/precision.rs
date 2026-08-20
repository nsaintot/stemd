//! Choosing a precision per model, rather than per run.
//!
//! `--full-precision` used to answer this for the whole server, which cannot be
//! right for a preset that chains two architectures: on CUDA a BS-RoFormer is
//! 3.76x faster at full precision and a demucs is 22.8x faster at half. Measured
//! end to end on Quality, a global switch to full precision is worth 1.16x; per
//! model it is worth about 3.5x on the model time.
//!
//! So the flag is an override and the default is a question asked once per model,
//! of [`stemd_core::Precision::preferred`].

use stemd_core::{Accelerator, Family, Precision};

use crate::models::Preset;

/// How this run decides what each model should run at.
///
/// Cheap to copy and stable for the life of the process: the accelerator is
/// detected once, because it cannot change under a running server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Precisions {
    /// `--full-precision`, which forces every model regardless of what suits it.
    forced: Option<Precision>,
    on: Accelerator,
}

impl Precisions {
    /// Ask MLX what it is running on, and take the flag as an override.
    pub fn detect(forced: Option<Precision>) -> Self {
        Self {
            forced,
            on: Accelerator::detect(),
        }
    }

    /// A stated answer rather than this machine's, so a test can assert
    /// something that does not depend on where it runs.
    #[cfg(test)]
    pub const fn stated(forced: Option<Precision>, on: Accelerator) -> Self {
        Self { forced, on }
    }

    pub const fn accelerator(self) -> Accelerator {
        self.on
    }

    pub const fn of(self, family: Family) -> Precision {
        match self.forced {
            Some(p) => p,
            None => Precision::preferred(family, self.on),
        }
    }

    /// What a preset's models will run at, in the order the preset uses them.
    ///
    /// One entry per model, so the identity below can tell "both halves at half
    /// precision" from "one at each" without knowing anything about presets.
    pub fn for_preset(self, preset: Option<Preset>) -> Vec<Precision> {
        Preset::families(preset)
            .iter()
            .map(|f| self.of(*f))
            .collect()
    }

    /// The precisions of a preset as one field of a cache key.
    ///
    /// Collapses to a single value when every model agrees, which is what keeps
    /// existing keys valid: on Metal, and on a CPU, they always do. Only a
    /// backend that wants different answers from different models produces the
    /// joined form, and there the audio really is different.
    pub fn key(self, preset: Option<Preset>) -> String {
        let all = self.for_preset(preset);
        match all.split_first() {
            Some((first, rest)) if rest.iter().all(|p| p == first) => first.to_string(),
            _ => all
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("+"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(a: Accelerator) -> Precisions {
        Precisions::stated(None, a)
    }

    /// The whole point: Quality's two halves disagree on CUDA and nowhere else.
    #[test]
    fn only_cuda_splits_a_chained_preset() {
        let q = Some(Preset::Quality);
        assert_eq!(on(Accelerator::Metal).for_preset(q), [Precision::F16; 2]);
        assert_eq!(on(Accelerator::Cpu).for_preset(q), [Precision::F32; 2]);
        assert_eq!(
            on(Accelerator::Cuda).for_preset(q),
            [Precision::F32, Precision::F16],
            "on CUDA the RoFormer half wants full and the demucs half wants half"
        );
    }

    /// A key that has not changed is a cache that survives. Metal and CPU keep
    /// the compact form they have always had.
    #[test]
    fn the_key_only_splits_when_the_models_do() {
        let q = Some(Preset::Quality);
        assert_eq!(on(Accelerator::Metal).key(q), "f16");
        assert_eq!(on(Accelerator::Cpu).key(q), "f32");
        assert_eq!(on(Accelerator::Cuda).key(q), "f32+f16");
    }

    /// A preset that is all one architecture never splits, on any backend.
    #[test]
    fn a_single_family_preset_never_splits() {
        for a in [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu] {
            for preset in [Some(Preset::Fast), Some(Preset::Balanced), None] {
                let key = on(a).key(preset);
                assert!(!key.contains('+'), "{a} {preset:?} split into {key}");
            }
        }
    }

    /// The flag still means what it says, everywhere.
    #[test]
    fn forcing_full_precision_overrides_every_model() {
        let forced = Precisions::stated(Some(Precision::F32), Accelerator::Cuda);
        assert_eq!(
            forced.for_preset(Some(Preset::Quality)),
            [Precision::F32; 2]
        );
        assert_eq!(forced.key(Some(Preset::Quality)), "f32");
    }
}
