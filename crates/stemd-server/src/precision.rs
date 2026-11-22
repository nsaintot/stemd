//! Chooses a precision for each model.
//!
//! One precision per model family: on CUDA a BS-RoFormer is 3.76x faster at full precision
//! and a demucs 22.8x faster at half, so a single switch for the whole server cannot suit a
//! preset that runs both.
//!
//! The flag is an override; by default, each model's precision comes from
//! [`stemd_core::Precision::preferred`], asked once per model.

use stemd_core::{Accelerator, Family, Precision};

use crate::models::Preset;

/// How this run decides what each model should run at.
///
/// Cheap to copy and stable for the life of the process: the accelerator is
/// detected once, because it cannot change under a running server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Precisions {
    /// `--full-precision`, which forces a precision on every model.
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

    /// A fixed accelerator, so a test does not depend on the machine it runs on.
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

    /// What a preset's models will run at, one entry per model in preset order.
    fn for_preset(self, preset: Option<Preset>) -> Vec<Precision> {
        Preset::families(preset)
            .iter()
            .map(|f| self.of(*f))
            .collect()
    }

    /// The precisions of a preset as one cache-key field: one value when all models agree, as on
    /// Metal and on a CPU, joined with `+` otherwise.
    ///
    /// The single-value form keeps existing cache keys valid.
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

    /// Quality is one RoFormer: half on Metal, full on CUDA and the CPU.
    #[test]
    fn quality_runs_at_what_a_roformer_wants() {
        let q = Some(Preset::Quality);
        assert_eq!(on(Accelerator::Metal).for_preset(q), [Precision::F16]);
        assert_eq!(on(Accelerator::Cpu).for_preset(q), [Precision::F32]);
        assert_eq!(on(Accelerator::Cuda).for_preset(q), [Precision::F32]);
        assert_eq!(on(Accelerator::Metal).key(q), "f16");
        assert_eq!(on(Accelerator::Cuda).key(q), "f32");
    }

    /// A preset that is all one architecture never splits, on any backend.
    #[test]
    fn a_single_family_preset_never_splits() {
        for a in [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu] {
            for preset in Preset::ALL.map(Some).into_iter().chain([None]) {
                let key = on(a).key(preset);
                assert!(!key.contains('+'), "{a} {preset:?} split into {key}");
            }
        }
    }

    /// The flag forces its precision on every backend.
    #[test]
    fn forcing_full_precision_overrides_every_model() {
        let forced = Precisions::stated(Some(Precision::F32), Accelerator::Cuda);
        assert_eq!(forced.for_preset(Some(Preset::Quality)), [Precision::F32]);
        assert_eq!(forced.key(Some(Preset::Quality)), "f32");
    }
}
