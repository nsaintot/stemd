//! Fetching the CUDA runtime, for a machine that has a card but not the libraries.
//!
//! The binary delay-loads its CUDA imports and decides at startup whether they can
//! be resolved, so it runs anywhere and uses a GPU only where one is usable. This
//! is the way across for a machine with a card running on its CPU: about 1.2 GB,
//! once, from NVIDIA's own redistributable archives.
//!
//! Windows only. Elsewhere CUDA is linked the ordinary way and a missing library
//! is a process that does not start.
//!
//! Not automatic: `startup` says the runtime is missing and names the flag, and
//! the flag is the whole of the consent.
//!
//! ## The versions here
//!
//! Pinned to CUDA 13.3.1 and cuDNN 9.14.0, with digests taken from the manifests
//! NVIDIA publishes them under:
//!
//! ```text
//! developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json
//! developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.14.0.json
//! ```
//!
//! 13.3 because that is the toolkit `mlx-sys` builds against, so the DLL names
//! here are the ones in the import table. They carry only a major version
//! (`cublasLt64_13`, `cufft64_12`, `nvrtc64_130_0`, `cudnn64_9`), which is what
//! makes a redist build usable against a binary linked elsewhere in the same major
//! series.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::models::RemoteFile;

/// What to fetch, and nothing else.
///
/// stemd does matmul, transforms and convolution; the archives for the rest of the
/// toolkit hold nothing this binary can call, 2404 MB of which 1090 MB is
/// reachable. `cuda_cudart` is here despite not being imported: it is 2.6 MB and
/// several of the others load it themselves.
pub const COMPONENTS: &[RemoteFile] = &[
    RemoteFile {
        name: "libcublas.zip",
        url: concat!(
            "https://developer.download.nvidia.com/compute/cuda/redist/",
            "libcublas/windows-x86_64/libcublas-windows-x86_64-13.6.0.2-archive.zip"
        ),
        sha256: "62e9fa30560c8f0a28e0cdcf9d6fc1fed347bcfab8847239b9ae1fdc1d86408a",
        bytes: 393_706_755,
    },
    RemoteFile {
        name: "libcufft.zip",
        url: concat!(
            "https://developer.download.nvidia.com/compute/cuda/redist/",
            "libcufft/windows-x86_64/libcufft-windows-x86_64-12.3.0.29-archive.zip"
        ),
        sha256: "83df908ae67e2b3a86201de8463562ab49dd9ee8b3b5efc3fdc2e681b14b5dd9",
        bytes: 182_627_436,
    },
    RemoteFile {
        name: "cudnn.zip",
        url: concat!(
            "https://developer.download.nvidia.com/compute/cudnn/redist/",
            "cudnn/windows-x86_64/cudnn-windows-x86_64-9.14.0.64_cuda13-archive.zip"
        ),
        sha256: "9b98b51bcead704e32640eca1770cadecec1bdf67c6db4093ff4fbcc4a206bd8",
        bytes: 334_801_105,
    },
    RemoteFile {
        name: "cuda_nvrtc.zip",
        url: concat!(
            "https://developer.download.nvidia.com/compute/cuda/redist/",
            "cuda_nvrtc/windows-x86_64/cuda_nvrtc-windows-x86_64-13.3.33-archive.zip"
        ),
        sha256: "8519f678588610bf380ccaac130729aa1a624c407183e7ad9c319c19ecc63d2f",
        bytes: 312_126_196,
    },
    RemoteFile {
        name: "cuda_cudart.zip",
        url: concat!(
            "https://developer.download.nvidia.com/compute/cuda/redist/",
            "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-13.3.29-archive.zip"
        ),
        sha256: "1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c",
        bytes: 2_589_792,
    },
];

/// Where the libraries have to end up: beside the executable.
///
/// The ordinary search order looks there first, which is where the probe looks and
/// where the delay-load helper falls back to. MLX's own helper also knows a
/// `../nvidia/<component>/bin` layout, which would work for MLX and not for the
/// probe.
pub fn beside_the_executable() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("finding this executable")?;
    Ok(exe
        .parent()
        .context("this executable has no directory")?
        .to_path_buf())
}

/// Fetch every component and put its libraries beside the executable.
///
/// `on_progress` is called with `(component, bytes_done, bytes_total)`, the same
/// shape the model download reports, so one progress bar serves both.
pub fn install(into: &Path, mut on_progress: impl FnMut(&str, u64, u64)) -> Result<usize> {
    if !into.is_dir() {
        bail!("{} is not a directory", into.display());
    }
    writable(into)?;

    let staging = into.join("cuda-download");
    std::fs::create_dir_all(&staging).with_context(|| format!("creating {}", staging.display()))?;

    let total: u64 = COMPONENTS.iter().map(|c| c.bytes).sum();
    tracing::info!(
        "installing the CUDA runtime into {} ({:.1} GB to download)",
        into.display(),
        total as f64 / 1e9
    );

    let mut placed = 0;
    for component in COMPONENTS {
        crate::models::fetch_file(&staging, component, &mut on_progress)?;
        let archive = staging.join(component.name);
        placed += unpack_libraries(&archive, into)
            .with_context(|| format!("unpacking {}", component.name))?;
        // The archive is a third of a gigabyte and its contents are now where
        // they are wanted. Keeping it would double the cost of the install for
        // the sake of a re-run that a re-download does just as well.
        std::fs::remove_file(&archive).ok();
    }
    std::fs::remove_dir(&staging).ok();

    tracing::info!("{placed} libraries installed; restart to use the GPU");
    Ok(placed)
}

/// Copy every DLL out of a redistributable archive, flattening the paths.
///
/// NVIDIA lays these out as `<component>-archive/bin/<name>.dll`. Only the DLLs
/// are wanted, and only in one directory, since a nested one would be outside the
/// search order. Matched on the extension: nothing else here ends in `.dll`.
fn unpack_libraries(archive: &Path, into: &Path) -> Result<usize> {
    let file =
        std::fs::File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading {} as a zip", archive.display()))?;

    let mut placed = 0;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        if !entry.is_file() {
            continue;
        }
        // `enclosed_name` refuses absolute paths and `..`, which is what stops
        // an archive writing outside the directory it was pointed at.
        let Some(path) = entry.enclosed_name() else {
            continue;
        };
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }

        let partial = into.join(format!("{name}.part"));
        let mut out = std::fs::File::create(&partial)
            .with_context(|| format!("creating {}", partial.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("writing {}", partial.display()))?;
        drop(out);
        // Renamed into place only once whole, so an interrupted install cannot
        // leave a truncated library for the probe to load successfully.
        std::fs::rename(&partial, into.join(name)).with_context(|| format!("publishing {name}"))?;
        tracing::debug!("installed {name}");
        placed += 1;
    }
    Ok(placed)
}

/// Fail on a read-only install directory before spending a gigabyte finding out.
///
/// Under `Program Files` this is the ordinary case rather than a strange one,
/// and the useful thing to say is which directory and that it needs elevation,
/// not a permission error a gigabyte later with a component name attached.
fn writable(dir: &Path) -> Result<()> {
    let probe = dir.join(".stemd-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            std::fs::remove_file(&probe).ok();
            Ok(())
        }
        Err(e) => bail!(
            "cannot write to {}: {e}. The CUDA libraries have to sit beside the \
             executable, so this needs a writable install directory or an \
             elevated prompt",
            dir.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two hosts a component may come from. Here rather than beside
    /// [`COMPONENTS`] because the URLs are written out in full: `concat!` takes
    /// literals, not constants, and a prefix that only the assertion below uses
    /// is dead code in every real build.
    const CUDA: &str = "https://developer.download.nvidia.com/compute/cuda/redist/";
    const CUDNN: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/";

    /// Every pinned digest is a sha256 and every URL is NVIDIA's own. A typo in
    /// either is a download that fails late, after the bytes have been paid for.
    #[test]
    fn the_pinned_components_are_well_formed() {
        assert!(!COMPONENTS.is_empty());
        for c in COMPONENTS {
            assert_eq!(c.sha256.len(), 64, "{}: not a sha256", c.name);
            assert!(
                c.sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
                "{}: not hex",
                c.name
            );
            assert!(
                c.url.starts_with(CUDA) || c.url.starts_with(CUDNN),
                "{}: {} is not an NVIDIA redist URL",
                c.name,
                c.url
            );
            assert!(c.url.ends_with(".zip"), "{}: not an archive", c.name);
            assert!(c.bytes > 1 << 20, "{}: implausibly small", c.name);
            assert!(
                c.name.ends_with(".zip"),
                "{}: staged under a bare name",
                c.name
            );
        }
    }

    /// The four libraries the binary delay-imports have to be covered by
    /// something here, or the install completes and changes nothing.
    #[test]
    fn every_delay_imported_library_has_a_component() {
        for (library, component) in [
            ("cublasLt64_13.dll", "libcublas.zip"),
            ("cufft64_12.dll", "libcufft.zip"),
            ("cudnn64_9.dll", "cudnn.zip"),
            ("nvrtc64_130_0.dll", "cuda_nvrtc.zip"),
        ] {
            assert!(
                COMPONENTS.iter().any(|c| c.name == component),
                "{library} would come from {component}, which is not fetched"
            );
        }
    }

    /// Distinct names, since they share one staging directory.
    #[test]
    fn no_two_components_stage_under_the_same_name() {
        let mut names: Vec<_> = COMPONENTS.iter().map(|c| c.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before);
    }

    /// A directory that cannot be written to is refused before anything is
    /// downloaded, and says which one.
    #[test]
    fn an_unwritable_target_is_refused_by_name() {
        let missing = Path::new("/definitely/not/a/directory/here");
        let err = writable(missing).unwrap_err().to_string();
        assert!(err.contains("not/a/directory"), "{err}");
    }
}
