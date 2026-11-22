//! Building a separator from a model artefact on disk.

use std::path::Path;

use anyhow::{Context, Result};
use stemd_core::{
    HybridConfig, HybridSeparator, MlxConfig, MlxSeparator, RoformerSeparator, Separate,
};

use super::Preset;
use crate::precision::Precisions;

/// How a separator reaches the worker: as a recipe, never as a finished model.
///
/// MLX weights must be allocated on the thread that will evaluate them. The CUDA
/// backend keeps its stream registry in thread-local storage, so a model built on
/// one thread cannot be evaluated on another. Handing over a closure rather than
/// a built model makes that unrepresentable. See [`stemd_core::mlx`].
pub type BuildSeparator = Box<dyn FnOnce() -> Result<Box<dyn Separate>> + Send>;

/// [`build`] as a recipe, for whichever thread is going to run the model.
///
/// The arguments are copied rather than borrowed because the closure crosses a
/// thread boundary. Both callers that install a model, the launch and the switch,
/// go through here. See [`BuildSeparator`].
pub fn builder(
    dir: &Path,
    artefact: &str,
    preset: Option<Preset>,
    overlap: f32,
    precisions: Precisions,
) -> BuildSeparator {
    let (dir, artefact) = (dir.to_path_buf(), artefact.to_owned());
    Box::new(move || {
        build(&dir, &artefact, preset, overlap, precisions)
            .with_context(|| format!("loading {artefact} from {}", dir.display()))
    })
}

/// Turn a directory and an artefact name into a loaded model.
///
/// Private, and reached only through [`builder`].
///
/// Which architecture is decided by the preset rather than by sniffing the file,
/// because only a preset can say that two artefacts belong together.
fn build(
    dir: &Path,
    artefact: &str,
    preset: Option<Preset>,
    overlap: f32,
    precisions: Precisions,
) -> Result<Box<dyn Separate>> {
    // Precision is asked per model family; see `crate::precision`.
    let families = Preset::families(preset);
    match preset {
        Some(Preset::Quality) => Ok(Box::new(RoformerSeparator::load(
            dir,
            Preset::Quality.artefact(),
            precisions.of(families[0]),
            overlap,
        )?)),
        // Two of the artefact's four models, unchained: chaining the demucs
        // vocals specialist into the drums half separates worse.
        Some(Preset::Balanced) => Ok(Box::new(HybridSeparator::demucs_specialists(
            dir,
            artefact,
            HybridConfig::new(
                overlap,
                precisions.of(families[0]),
                precisions.of(families[0]),
            ),
        )?)),
        _ => Ok(Box::new(MlxSeparator::load(
            dir,
            artefact,
            MlxConfig {
                overlap,
                precision: precisions.of(families[0]),
            },
        )?)),
    }
}
