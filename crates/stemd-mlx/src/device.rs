//! Which backend MLX will run on, and whether it can actually do the work.
//!
//! Three values rather than two, because Metal and CUDA do not want the same
//! precision from the same model. See [`crate::Precision::preferred`].
//!
//! Asking MLX which device it picked is not enough. MLX seeds that from
//! `gpu::is_available()`, which on CUDA is satisfied by the driver alone, so a
//! machine carrying `nvcuda.dll` and none of cuBLAS, cuFFT or cuDNN answers yes
//! and then dies on the first separation. Removing that is what makes a Windows
//! build shippable without the CUDA runtime beside it: the binary delay-loads
//! those DLLs, so it starts anywhere, and the answer here decides whether it uses
//! them.
//!
//! So the decision is taken in two steps, in this order:
//!
//! 1. Can the CUDA runtime be resolved at all? A delay-load stub that cannot find
//!    its DLL raises a structured exception, which kills the process, so this has
//!    to be settled before MLX is asked anything.
//! 2. Does a GPU produce right answers? Matmul, both transforms and a
//!    convolution, evaluated and checked, which is every kind of work the two
//!    models are built from.
//!
//! Not a device selection: nothing here picks a GPU. It picks the CPU, and only
//! ever as a demotion.

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

/// What [`Accelerator::decide`] worked out, kept because the probe costs a
/// handful of GPU operations and `detect` is called per request.
static DECIDED: OnceLock<(Accelerator, bool)> = OnceLock::new();

impl Accelerator {
    /// What MLX will actually use.
    ///
    /// Decided once, on the first call, and enforced: if this answers [`Self::Cpu`]
    /// then MLX's own default device has been set to the CPU too. Which GPU is a
    /// build-time fact: Metal on Apple, CUDA everywhere else.
    pub fn detect() -> Self {
        DECIDED.get_or_init(Self::decide).0
    }

    /// True when a GPU was there and this module turned it down.
    ///
    /// The two kinds of [`Self::Cpu`] want different things said: a machine without a
    /// card is nothing to report, while a card whose runtime is missing needs telling,
    /// because the fix is a download.
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
        if !gpu_computes() {
            return (Self::demote(), true);
        }
        let which = if cfg!(target_vendor = "apple") {
            Self::Metal
        } else {
            Self::Cuda
        };
        (which, false)
    }

    /// Point MLX at the CPU as well as saying so.
    fn demote() -> Self {
        Device::set_default(&Device::cpu());
        Self::Cpu
    }

    /// The name this carries over the wire, in `BackendInfo::device`. Stable,
    /// and deliberately coarse: clients match on it, and none of them has any
    /// use for which GPU it is.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal | Self::Cuda => "gpu",
            Self::Cpu => "cpu",
        }
    }

    /// The backend by name, for logs and for anything reasoning about the
    /// difference between the two GPUs.
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

/// Whether the GPU MLX offered returns right answers.
///
/// One operation of each kind the models use, on inputs small enough that the
/// expected sums can be written down: matmul, the real transform both ways, and a
/// convolution. Each is evaluated, because MLX is lazy.
///
/// Values are checked rather than only the absence of an error: a defect this port
/// found returned a number, and it was 65 times too large.
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

/// Whether the CUDA libraries this build imports can be found.
///
/// Only Windows needs this, because Windows is the platform where the runtime is
/// optional. `mlx-sys` emits one `/DELAYLOAD` per CUDA DLL, so none is bound at
/// process start and a build with no CUDA installed starts cleanly. The cost is
/// that a missing one surfaces as a structured exception at the first call, which
/// nothing can catch usefully, so the libraries are looked for before MLX is
/// asked anything and a miss means the CPU.
///
/// Elsewhere they are ordinary imports resolved before `main`, so a missing one
/// is a process that does not start. Nothing to check.
#[cfg(all(windows, target_env = "msvc"))]
mod runtime {
    use std::ffi::c_void;

    /// The driver. Installed by the display driver rather than by the toolkit,
    /// so it is present on any machine with an NVIDIA card and absent on any
    /// machine without one, which is the question `gpu::is_available()` is
    /// really answering. It is not imported, hard or delayed, so it has to be
    /// named here.
    const DRIVER: &str = "nvcuda.dll";

    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *const u8;
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn FreeLibrary(module: *mut c_void) -> i32;
    }

    /// Whether every delay-loaded library can be found.
    ///
    /// This executable has no CUDA in its ordinary imports at all, so the
    /// delay-import directory is the whole of what can fault, and loading all
    /// of it is the exact statement that nothing will.
    pub fn resolves() -> bool {
        delay_loaded().iter().all(|name| loads(name))
    }

    pub fn driver_present() -> bool {
        loads(DRIVER)
    }

    /// The ordinary search order, which the delay-load helper uses too: beside the
    /// executable first, then `PATH`. A load rather than a test for the file, so a
    /// library whose own dependencies are missing counts as missing.
    ///
    /// MLX's helper also tries the toolkit directory from the machine that built it,
    /// so a developer tree whose DLLs are neither staged nor on `PATH` runs on the CPU.
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
    /// Read out of the image rather than kept as a list. `build.rs` offers the linker
    /// 36 candidates and the linker keeps whichever are genuinely imported, so a
    /// hand-written list would be wrong in both directions: naming libraries the
    /// binary never calls, and going stale when MLX or the toolkit changes which it
    /// does.
    ///
    /// An empty answer means nothing is delay-loaded and so nothing can fault, which
    /// is the correct answer for a build without CUDA rather than a failure to look.
    fn delay_loaded() -> Vec<String> {
        /// Where the PE header offset sits in the DOS header.
        const PE_OFFSET: usize = 0x3c;
        /// Signature and file header, before the optional header.
        const NT_TO_OPTIONAL: usize = 24;
        /// Within a PE32+ optional header: the data directories, and the count
        /// of them that precedes them.
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
                // The table ends in a descriptor of zeroes. Bit 0 of the
                // attributes says the addresses in it are RVAs, which every
                // linker since VC7 sets; without it this is an image from
                // before delay loading looked like this, and guessing is worse
                // than stopping.
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

    /// Nothing asserts *which* libraries these are: that is the build's
    /// business and this module's whole point is not to hold an opinion about
    /// it. What can be checked anywhere is that the walk terminates, agrees
    /// with itself, and returns plausible names rather than rubble.
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

    /// Whichever this machine is, the coarse name is one of the two a client
    /// knows, and the two GPUs both report as one.
    #[test]
    fn the_wire_name_stays_coarse() {
        assert_eq!(Accelerator::Metal.as_str(), "gpu");
        assert_eq!(Accelerator::Cuda.as_str(), "gpu");
        assert_eq!(Accelerator::Cpu.as_str(), "cpu");
        assert!(matches!(Accelerator::detect().as_str(), "gpu" | "cpu"));
    }

    /// The backend name is the one that distinguishes them, and it is what
    /// picks the precision.
    #[test]
    fn the_backend_name_does_not() {
        assert_eq!(Accelerator::Metal.backend(), "metal");
        assert_eq!(Accelerator::Cuda.backend(), "cuda");
    }

    /// The answer is enforced rather than reported. This is the whole point of
    /// the probe: a build that says `cpu` while MLX keeps sending work to a
    /// GPU it cannot use has moved the failure, not removed it.
    #[test]
    fn mlx_is_left_agreeing_with_the_answer() {
        let mine = Accelerator::detect();
        let mlx = Device::try_default().and_then(|d| d.get_type());
        match mine {
            Accelerator::Cpu => assert!(matches!(mlx, Ok(DeviceType::Cpu))),
            Accelerator::Metal | Accelerator::Cuda => {
                assert!(matches!(mlx, Ok(DeviceType::Gpu)));
            }
        }
    }

    /// Taken once. Callers ask per request, and the probe is not free.
    #[test]
    fn the_decision_does_not_change_under_it() {
        assert_eq!(Accelerator::detect(), Accelerator::detect());
    }

    /// A refusal needs a GPU to refuse. Whatever this machine is, the two
    /// answers cannot both be "there is a card and we are using it" and "there
    /// is a card and we turned it down".
    #[test]
    fn a_working_gpu_is_never_reported_as_refused() {
        if Accelerator::detect() != Accelerator::Cpu {
            assert!(!Accelerator::gpu_refused());
        }
    }

    /// The sums the probe checks against are the right ones.
    ///
    /// Deliberately not `gpu_computes() == (detect() != Cpu)`: the probe runs on MLX's
    /// default device, so on a machine with no GPU it passes on the CPU. What it can
    /// prove anywhere is that a healthy backend clears it.
    #[test]
    fn the_probe_arithmetic_is_right() {
        Accelerator::detect();
        assert!(gpu_computes());
    }
}
