//! Builds a separator from a model artefact on disk.

use std::path::Path;

use anyhow::{Context, Result};
use stemd_core::{
    HybridConfig, HybridSeparator, MlxConfig, MlxSeparator, RoformerSeparator, Separate,
};

use super::Preset;
use crate::precision::Precisions;

/// How a separator reaches the worker: as a closure that builds it, never as a built model.
// MLX weights must be allocated on the thread that evaluates them: the CUDA backend keeps its
// stream registry in thread-local storage, so a model built on one thread cannot run on another.
// Passing a closure makes that mistake impossible to write. See [`stemd_core::mlx`].
pub type BuildSeparator = Box<dyn FnOnce() -> Result<Box<dyn Separate>> + Send>;

/// [`build`] wrapped in a closure, for the thread that will run the model.
// The arguments are copied because the closure crosses a thread boundary. Both callers that
// install a model, the launch and the switch, go through here. See [`BuildSeparator`].
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

/// Load the model for an artefact name in a directory. Reached only through [`builder`].
///
/// The preset decides the architecture, because only a preset can say that two artefacts belong
/// together.
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
        // Run two of the artefact's four models side by side: feeding the demucs vocals
        // specialist's output into the drums half separates worse.
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
