//! Which backend MLX runs on, and whether it can do the work.
//!
//! There are three values because Metal and CUDA want different precisions from the same model. See
//! [`crate::Precision::preferred`].
//!
//! Asking MLX which device it picked is not enough. MLX bases that on `gpu::is_available()`, which
//! on CUDA needs only the driver, so a machine with `nvcuda.dll` and none of cuBLAS, cuFFT or cuDNN
//! answers yes and then dies on the first separation. The checks here let a Windows build ship
//! without the CUDA runtime beside it: the binary delay-loads those DLLs, so it starts anywhere,
//! and the answer here decides whether it uses them.
//!
//! The decision takes two steps, in this order:
//!
//! 1. Can the CUDA runtime be resolved at all? A delay-load stub that cannot find
//!    its DLL raises a structured exception, which kills the process, so this is
//!    settled before MLX is asked anything.
//! 2. Does the GPU produce right answers? A matmul, both transforms and a
//!    convolution are evaluated and checked; together they cover every kind of
//!    work the two models do.
//!
//! Nothing here selects a GPU. This module can only move the run to the CPU.

use std::sync::OnceLock;

use anyhow::Result;
use mlx_rs::{Array, Device, DeviceType, ops};

/// Where MLX will run the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Accelerator {
    /// A GPU on Apple silicon.
    Metal,
    /// A GPU anywhere else.
    Cuda,
    /// No GPU in this build, none on this machine, or one that failed the
    /// probe.
    Cpu,
}

/// The result of [`Accelerator::decide`], cached because the probe costs a handful of GPU
/// operations and `detect` is called per request.
static DECIDED: OnceLock<(Accelerator, bool)> = OnceLock::new();

impl Accelerator {
    /// The backend MLX uses.
    ///
    /// It is decided once, on the first call, and enforced: when this returns [`Self::Cpu`], MLX's
    /// own default device is also set to the CPU. The GPU kind is fixed at build time: Metal on
    /// Apple, CUDA everywhere else.
    pub fn detect() -> Self {
        DECIDED.get_or_init(Self::decide).0
    }

    /// True when a GPU was present and this module refused it.
    ///
    /// The two kinds of [`Self::Cpu`] need different messages: a machine without a card needs none,
    /// while a card whose runtime is missing needs a warning, because a download fixes it.
    pub fn gpu_refused() -> bool {
        DECIDED.get_or_init(Self::decide).1
    }

    fn decide() -> (Self, bool) {
        if !runtime::resolves() {
            return (Self::demote(), runtime::driver_present());
        }
        let gpu = matches!(
            Device::try_default().and_then(|d| d.get_type()),
            Ok(DeviceType::Gpu)
        );
        if !gpu {
            return (Self::Cpu, false);
        }
        if !gpu_computes_on_kept_thread() {
            return (Self::demote(), true);
        }
        let which = if cfg!(target_vendor = "apple") {
            Self::Metal
        } else {
            Self::Cuda
        };
        (which, false)
    }

    /// Point MLX at the CPU and return [`Self::Cpu`].
    fn demote() -> Self {
        Device::set_default(&Device::cpu());
        Self::Cpu
    }

    /// The name sent to clients in `BackendInfo::device`. It is stable and coarse on purpose:
    /// clients match on it, and none needs to know which GPU it is.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal | Self::Cuda => "gpu",
            Self::Cpu => "cpu",
        }
    }

    /// The backend by name, for logs and for code that tells the two GPUs apart.
    pub const fn backend(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

impl std::fmt::Display for Accelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.backend())
    }
}

/// [`gpu_computes`] on a thread that never exits.
///
/// MLX keeps CUDA state per thread, and on Windows its destructor throws when the
/// thread exits. Windows reports that as a crash and freezes the whole process for
/// about 3 s while it writes the report.
fn gpu_computes_on_kept_thread() -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("stemd-gpu-probe".into())
        .spawn(move || {
            let _ = tx.send(gpu_computes());
            loop {
                std::thread::park();
            }
        });
    spawned.is_ok() && rx.recv().unwrap_or(false)
}

/// Whether the GPU that MLX offered returns right answers.
// It runs one operation of each kind the models use, on inputs small enough that the expected sums
// can be written down: matmul, the real transform both ways, and a convolution. Each is evaluated,
// because MLX is lazy.
//
// The values are checked as well as the absence of an error, because a broken kernel can return
// wrong values without failing.
fn gpu_computes() -> bool {
    fn probe() -> Result<bool> {
        let square = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let product = ops::matmul(&square, &square)?;

        let wave = Array::from_slice(&[1.0f32; 8], &[8]);
        let spectrum = mlx_rs::fft::rfft(&wave, None, -1)?;
        let restored = mlx_rs::fft::irfft(&spectrum, 8, -1)?;

        let signal = Array::from_slice(&[1.0f32; 8], &[1, 8, 1]);
        let kernel = Array::from_slice(&[1.0f32; 3], &[1, 3, 1]);
        let filtered = ops::conv1d(&signal, &kernel, 1, 0, 1, 1)?;

        mlx_rs::transforms::eval([&product, &restored, &filtered])?;

        let near = |value: f32, want: f32| (value - want).abs() < 1e-3;
        Ok(near(ops::sum(&product, None)?.item::<f32>(), 54.0)
            && near(ops::sum(&restored, None)?.item::<f32>(), 8.0)
            && near(ops::sum(&filtered, None)?.item::<f32>(), 18.0))
    }
    probe().unwrap_or(false)
}

/// Whether the CUDA libraries this build imports can be found. Only Windows needs this, because
/// only there is the runtime optional.
// `mlx-sys` emits one `/DELAYLOAD` per CUDA DLL, so none is bound at process start and a build with
// no CUDA installed starts cleanly. In exchange, a missing library surfaces as a structured
// exception at the first call, which nothing can usefully catch, so the libraries are looked for
// before MLX is asked anything, and a miss means the CPU.
//
// On other platforms they are ordinary imports resolved before `main`, so a missing one stops the
// process from starting, and there is nothing to check.
#[cfg(all(windows, target_env = "msvc"))]
mod runtime {
    use std::ffi::c_void;

    /// The driver. The display driver installs it, and the CUDA toolkit does not, so it is present
    /// on every machine with an NVIDIA card and absent on every machine without one, which is what
    /// `gpu::is_available()` really tests. It is neither a hard nor a delayed import, so it is
    /// named here.
    const DRIVER: &str = "nvcuda.dll";

    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *const u8;
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    /// Whether every delay-loaded library can be found.
    ///
    /// This executable has no CUDA in its ordinary imports, so the delay-import directory is
    /// everything that can fault, and loading all of it proves that nothing will.
    pub fn resolves() -> bool {
        delay_loaded().iter().all(|name| loads(name))
    }

    pub fn driver_present() -> bool {
        loads(DRIVER)
    }

    /// Load `name` with the ordinary search order, which the delay-load helper also uses: next to
    /// the executable first, then `PATH`. Loading the library also fails when one of its own
    /// dependencies is missing.
    // MLX's helper also tries the toolkit directory of the machine that built it, so a developer
    // tree whose DLLs are neither staged nor on `PATH` runs on the CPU.
    fn loads(name: &str) -> bool {
        let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
        let module = unsafe { LoadLibraryW(wide.as_ptr()) };
        if module.is_null() {
            return false;
        }
        unsafe { FreeLibrary(module) };
        true
    }

    /// The names in this image's delay-import directory.
    ///
    /// They are read from the image: `build.rs` offers the linker 36 candidates and the linker
    /// keeps only those actually imported, so a hand-written list would be wrong both ways, naming
    /// libraries the binary never calls and going stale when MLX or the toolkit changes its
    /// imports.
    ///
    /// An empty answer means nothing is delay-loaded, so nothing can fault; that is the correct
    /// answer for a build without CUDA.
    fn delay_loaded() -> Vec<String> {
        /// Where the PE header offset sits in the DOS header.
        const PE_OFFSET: usize = 0x3c;
        /// The size of the signature and the file header, before the optional header.
        const NT_TO_OPTIONAL: usize = 24;
        /// The offset of the data directories in a PE32+ optional header, just past their count.
        const OPTIONAL_TO_DIRECTORIES: usize = 112;
        const OPTIONAL_TO_DIRECTORY_COUNT: usize = 108;
        /// IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT.
        const DELAY_IMPORT: usize = 13;
        /// sizeof(ImgDelayDescr), and the offset of its `rvaDLLName`.
        const DESCRIPTOR: usize = 32;
        const NAME_RVA: usize = 4;

        let base = unsafe { GetModuleHandleW(std::ptr::null()) };
        if base.is_null() {
            return Vec::new();
        }
        // The image is mapped, so every RVA below is an offset from `base`, and
        // each read is inside a directory the loader has already validated.
        unsafe {
            if u16_at(base, 0) != 0x5a4d {
                return Vec::new(); // not "MZ"
            }
            let nt = u32_at(base, PE_OFFSET) as usize;
            if u32_at(base, nt) != 0x0000_4550 {
                return Vec::new(); // not "PE\0\0"
            }
            let optional = nt + NT_TO_OPTIONAL;
            if u16_at(base, optional) != 0x20b {
                return Vec::new(); // not PE32+, so the offsets below are wrong
            }
            if u32_at(base, optional + OPTIONAL_TO_DIRECTORY_COUNT) as usize <= DELAY_IMPORT {
                return Vec::new();
            }
            let entry = optional + OPTIONAL_TO_DIRECTORIES + DELAY_IMPORT * 8;
            let table = u32_at(base, entry) as usize;
            let bytes = u32_at(base, entry + 4) as usize;

            let mut names = Vec::new();
            let mut at = table;
            while table != 0 && at + DESCRIPTOR <= table + bytes {
                let name = u32_at(base, at + NAME_RVA) as usize;
                // The table ends in a descriptor of zeroes. Bit 0 of the attributes says the
                // addresses are RVAs, which every linker since VC7 sets; without it the image uses
                // the older delay-load layout, and stopping is safer than guessing.
                if name == 0 || u32_at(base, at) & 1 == 0 {
                    break;
                }
                names.push(ascii_at(base, name));
                at += DESCRIPTOR;
            }
            names
        }
    }

    unsafe fn u16_at(base: *const u8, offset: usize) -> u16 {
        unsafe { base.add(offset).cast::<u16>().read_unaligned() }
    }

    unsafe fn u32_at(base: *const u8, offset: usize) -> u32 {
        unsafe { base.add(offset).cast::<u32>().read_unaligned() }
    }

    /// A DLL name, which the format says is a null-terminated ASCII string.
    unsafe fn ascii_at(base: *const u8, offset: usize) -> String {
        unsafe {
            let start = base.add(offset);
            let mut len = 0;
            while *start.add(len) != 0 {
                len += 1;
            }
            String::from_utf8_lossy(std::slice::from_raw_parts(start, len)).into_owned()
        }
    }

    /// Nothing asserts which libraries these are, since that depends on the build and this module
    /// deliberately holds no list. What can be checked on any machine is that the walk terminates,
    /// agrees with itself, and returns plausible names.
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_delay_import_walk_reads_the_image_it_is_in() {
            let names = delay_loaded();
            assert_eq!(names, delay_loaded());
            for name in &names {
                assert!(name.len() > 4, "implausible import name {name:?}");
                assert!(name.to_ascii_lowercase().ends_with(".dll"), "{name:?}");
                assert!(name.is_ascii(), "{name:?}");
            }
        }
    }
}

#[cfg(not(all(windows, target_env = "msvc")))]
mod runtime {
    pub const fn resolves() -> bool {
        true
    }

    pub const fn driver_present() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On any machine, the coarse name is one of the two a client knows, and both GPUs report the
    /// same name.
    #[test]
    fn the_wire_name_stays_coarse() {
        assert_eq!(Accelerator::Metal.as_str(), "gpu");
        assert_eq!(Accelerator::Cuda.as_str(), "gpu");
        assert_eq!(Accelerator::Cpu.as_str(), "cpu");
        assert!(matches!(Accelerator::detect().as_str(), "gpu" | "cpu"));
    }

    /// The backend name tells the GPUs apart, and it picks the precision.
    #[test]
    fn the_backend_name_tells_the_gpus_apart() {
        assert_eq!(Accelerator::Metal.backend(), "metal");
        assert_eq!(Accelerator::Cuda.backend(), "cuda");
    }

    /// The answer is enforced as well as reported. A build that reports `cpu` while MLX keeps
    /// sending work to an unusable GPU would only move the failure.
    #[test]
    fn mlx_uses_the_device_that_detect_reports() {
        let mine = Accelerator::detect();
        let mlx = Device::try_default().and_then(|d| d.get_type());
        match mine {
            Accelerator::Cpu => assert!(matches!(mlx, Ok(DeviceType::Cpu))),
            Accelerator::Metal | Accelerator::Cuda => {
                assert!(matches!(mlx, Ok(DeviceType::Gpu)));
            }
        }
    }

    /// The accelerator is decided once, because callers ask per request and the probe runs GPU
    /// work.
    #[test]
    fn the_accelerator_is_decided_once() {
        assert_eq!(Accelerator::detect(), Accelerator::detect());
    }

    /// A GPU can be refused only if one was found, so on any machine "a card is in use" and "a card
    /// was refused" never both hold.
    #[test]
    fn a_working_gpu_is_never_reported_as_refused() {
        if Accelerator::detect() != Accelerator::Cpu {
            assert!(!Accelerator::gpu_refused());
        }
    }

    /// The sums the probe checks against are the right ones.
    ///
    /// The test does not assert `gpu_computes() == (detect() != Cpu)`: the probe runs on MLX's
    /// default device, so on a machine with no GPU it passes on the CPU. What it proves on any
    /// machine is that a healthy backend passes.
    #[test]
    fn the_probe_arithmetic_is_right() {
        Accelerator::detect();
        assert!(gpu_computes());
    }
}
