//! Delay-load the CUDA runtime into this executable, so it starts without one.
//!
//! `mlx-sys` emits these same flags, but `cargo:rustc-link-arg` applies only to the
//! package that emits it, so none of them reaches a consuming binary. Four of the
//! DLLs below would then be ordinary imports, bound before `main`, and a machine
//! without CUDA would get `0xC0000135` instead of a program.
//!
//! Delay-loaded, the same executable starts with none of them installed and reaches
//! for one only when a code path needs it, so CUDA is a download rather than a
//! precondition. Staged it is 134 MB: the exe, the MSVC runtime and OpenBLAS.
//!
//! What makes that safe is `stemd_mlx::Accelerator::detect`, which looks for these
//! libraries before MLX is asked anything and settles for the CPU when they are
//! missing. Without it a missing DLL becomes a server that accepts work and dies
//! doing it: the delay-load helper raises a structured exception nothing can
//! usefully catch, so the only defence is not to make the call.
//!
//! The list has to match `mlx-sys`'s, which writes its own copy to
//! `delayload_flags.txt` in its `OUT_DIR`. Cargo cannot hand it over without a
//! `links` key, so this is a copy and a toolkit bump changes both.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Delay loading is an MSVC linker feature. Everywhere else these are
    // ordinary shared libraries, resolved before `main`, and a missing one is a
    // process that does not start, which is already a clear failure.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    icon_and_version();
    for dll in DELAY_LOADED {
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:{dll}");
    }
    // delayimp.lib provides __delayLoadHelper2, which the stubs call.
    println!("cargo:rustc-link-arg-bins=delayimp.lib");
}

/// Give the executable its icon and its version tab.
///
/// The window sets its own icon at run time, which covers the taskbar and the
/// title bar. This is the other one: what Explorer, a Start menu shortcut and
/// Add/Remove Programs read, and none of them ever run the program. Without it
/// the installer's entry is a blank page beside every other application's mark.
#[cfg(windows)]
fn icon_and_version() {
    println!("cargo:rerun-if-changed=../../resources/stemd.ico");
    winresource::WindowsResource::new()
        .set_icon("../../resources/stemd.ico")
        .set("FileDescription", "stemd")
        .set("ProductName", "stemd")
        .set("LegalCopyright", "MIT OR Apache-2.0")
        .compile()
        .expect("embedding the icon and version resource");
}

/// Nothing to embed: the resource compiler is a Windows tool, and a build script
/// on another host has none. Reached only when cross-compiling to msvc, which
/// this project does not do, so it is silent rather than a warning.
#[cfg(not(windows))]
fn icon_and_version() {}

/// Every DLL `mlx-sys` delay-loads. The linker keeps the ones this binary imports
/// and answers `LNK4199` for the rest, which on MLX v0.31.2 is four kept and
/// thirty-two ignored. Trimming to those four would mean the next MLX to call
/// cuBLAS directly quietly acquires a hard import, which is a release that no
/// longer starts without CUDA.
const DELAY_LOADED: &[&str] = &[
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cudart64_13.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_ext64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cufft64_12.dll",
    "cufftw64_12.dll",
    "curand64_10.dll",
    "cusolver64_12.dll",
    "cusolverMg64_12.dll",
    "cusparse64_12.dll",
    "nppc64_13.dll",
    "nppial64_13.dll",
    "nppicc64_13.dll",
    "nppidei64_13.dll",
    "nppif64_13.dll",
    "nppig64_13.dll",
    "nppim64_13.dll",
    "nppist64_13.dll",
    "nppisu64_13.dll",
    "nppitc64_13.dll",
    "npps64_13.dll",
    "nvJitLink_130_0.dll",
    "nvblas64_13.dll",
    "nvfatbin_130_0.dll",
    "nvjpeg64_13.dll",
    "nvrtc-builtins64_133.dll",
    "nvrtc64_130_0.dll",
];
