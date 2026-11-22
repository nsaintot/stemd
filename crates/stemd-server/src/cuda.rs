//! Downloads the CUDA runtime, for a machine that has an NVIDIA GPU but not the libraries.
//!
//! The binary delay-loads its CUDA imports and decides at startup whether they can be resolved, so
//! it runs anywhere and uses a GPU only where one is usable. A machine with a GPU that runs on its
//! CPU gets the libraries here: about 1.2 GB, once, from NVIDIA's redistributable archives.
//!
//! Windows only. On other platforms CUDA is linked the ordinary way, and a missing library stops
//! the process from starting.
//!
//! The download never starts on its own: `startup` reports that the runtime is missing and names
//! the flag, and passing the flag is the user's consent.
//!
//! ## Versions
//!
//! Pinned to CUDA 13.3.1 and cuDNN 9.14.0, with digests taken from NVIDIA's manifests:
//!
//! ```text
//! developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json
//! developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.14.0.json
//! ```
//!
//! 13.3 is the toolkit `mlx-sys` builds against, so the DLL names here match the import table. The
//! names carry only a major version (`cublasLt64_13`, `cufft64_12`, `nvrtc64_130_0`, `cudnn64_9`),
//! so a redist build works with a binary linked against another release of the same major series.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::models::RemoteFile;

/// The components to download.
///
/// stemd uses matmul, transforms and convolution, and the rest of the toolkit holds nothing this
/// binary can call: the full archives total 2404 MB, of which these components are 1090 MB.
/// `cuda_cudart` is listed although the binary does not import it, because it is 2.6 MB and several
/// other components load it.
const COMPONENTS: &[RemoteFile] = &[
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

/// Point MLX's run-time kernel compiler at the CUDA headers shipped next to the executable, so a
/// machine without the toolkit can compile kernels.
pub fn point_at_bundled_headers() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(dir) = exe.parent() else { return };
    let cuda = dir.join("cuda");
    if cuda.join("include").join("cuda_runtime.h").is_file() {
        // SAFETY: called from main before any other thread exists.
        unsafe { std::env::set_var("CUDA_HOME", &cuda) };
    }
}

pub fn beside_the_executable() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("finding this executable")?;
    Ok(exe
        .parent()
        .context("this executable has no directory")?
        .to_path_buf())
}

/// Fetch every component and put its libraries next to the executable. The ordinary DLL search
/// order checks that directory first, and the probe and the delay-load helper look there.
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
        crate::models::fetch_one(&staging, component, &mut on_progress)?;
        let archive = staging.join(component.name);
        placed += unpack_libraries(&archive, into)
            .with_context(|| format!("unpacking {}", component.name))?;
        // Delete the archive, a third of a gigabyte, now that its libraries are in place; a re-run
        // downloads it again.
        std::fs::remove_file(&archive).ok();
    }
    std::fs::remove_dir(&staging).ok();

    tracing::info!("{placed} libraries installed; restart to use the GPU");
    Ok(placed)
}

/// Copy every DLL out of a redistributable archive, flattening the paths.
// NVIDIA lays them out as `<component>-archive/bin/<name>.dll`. All DLLs go into one directory,
// since a nested directory would be outside the search order. Entries are matched by the `.dll`
// extension, which no other file in these archives has.
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
        if !wanted(name) {
            continue;
        }

        let partial = into.join(format!("{name}.part"));
        let mut out = std::fs::File::create(&partial)
            .with_context(|| format!("creating {}", partial.display()))?;
        std::io::copy(&mut entry, &mut out)
            .with_context(|| format!("writing {}", partial.display()))?;
        drop(out);
        // Rename into place only once the file is whole, so an interrupted install cannot leave a
        // truncated library that the probe loads.
        crate::disk::publish(&partial, &into.join(name))?;
        tracing::debug!("installed {name}");
        placed += 1;
    }
    Ok(placed)
}

/// cuDNN libraries MLX never loads: it uses only the graph API, and no other
/// cuDNN library links these.
const UNUSED: &[&str] = &[
    "cudnn_adv64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_ext64_9.dll",
];

/// Whether an archive entry is a DLL to install.
fn wanted(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.ends_with(".dll") && !UNUSED.contains(&name.as_str())
}

/// Fail on a read-only install directory before downloading a gigabyte.
///
/// Under `Program Files` this is the normal case, so the error names the directory and says it
/// needs elevation.
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

    /// The two hosts a component may come from. They are defined here because the URLs in
    /// [`COMPONENTS`] are written out in full: `concat!` takes only literals, and a prefix used
    /// only by the assertion below would be dead code in every real build.
    const CUDA: &str = "https://developer.download.nvidia.com/compute/cuda/redist/";
    const CUDNN: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/";

    /// Every pinned digest is a SHA-256 and every URL is NVIDIA's own; a typo in either would fail
    /// only after the download.
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

    /// The components cover the four libraries the binary delay-imports; otherwise the install
    /// would complete and change nothing.
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

    #[test]
    fn the_unused_cudnn_libraries_are_skipped() {
        assert!(wanted("cudnn64_9.dll"));
        assert!(wanted("cudnn_graph64_9.dll"));
        assert!(wanted("cublasLt64_13.dll"));
        assert!(!wanted("cudnn_ops64_9.dll"));
        assert!(!wanted("CUDNN_ADV64_9.DLL"));
        assert!(!wanted("cudnn_cnn64_9.dll"));
        assert!(!wanted("cudnn_ext64_9.dll"));
        assert!(wanted("cudnn_engines_tensor_ir64_9.dll"));
        assert!(!wanted("cudnn.h"));
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
