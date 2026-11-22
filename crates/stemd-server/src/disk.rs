//! Free disk space and memory pressure, read so work is refused or trimmed
//! before the machine runs out.
//!
//! With the disk full, macOS cannot grow swap, and a separation that needs more
//! memory stalls the whole process inside an allocation.

use std::path::Path;

/// Free space below which new work is refused and caches stop growing.
pub const FLOOR: u64 = 2_000_000_000;

/// Bytes free to this user on the volume holding `path`, or `None` if it cannot
/// be read. A path that does not exist yet is measured at its nearest existing
/// ancestor.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let existing = path.ancestors().find(|p| p.exists())?;
    free_at(existing)
}

/// Whether the volume holding `path` is under [`FLOOR`].
pub fn low(path: &Path) -> bool {
    free_bytes(path).is_some_and(|free| free < FLOOR)
}

/// Work refused for want of disk space.
#[derive(Debug, thiserror::Error)]
#[error(
    "not enough disk space: {:.1} GB free, {:.1} GB needed; free some space and try again",
    *free as f64 / 1e9,
    *needed as f64 / 1e9
)]
pub struct DiskFull {
    pub free: u64,
    pub needed: u64,
}

/// Fail unless the volume holding `path` has [`FLOOR`] free. Passes when the
/// free space cannot be read.
pub fn ensure_room(path: &Path) -> Result<(), DiskFull> {
    ensure_room_for(path, 0)
}

/// Fail unless the volume holding `path` can take `bytes` and keep [`FLOOR`]
/// free.
pub fn ensure_room_for(path: &Path, bytes: u64) -> Result<(), DiskFull> {
    let needed = bytes.saturating_add(FLOOR);
    match free_bytes(path) {
        Some(free) if free < needed => Err(DiskFull { free, needed }),
        _ => Ok(()),
    }
}

/// Room a cache may hold: its cap, reduced so the volume keeps [`FLOOR`] free.
/// `used` is what the cache already holds on that volume.
pub fn budget(cap: u64, used: u64, path: &Path) -> u64 {
    free_bytes(path).map_or(cap, |free| fit(cap, used, free))
}

/// `cap`, cut so that a cache holding `used` with `free` left keeps [`FLOOR`] free.
const fn fit(cap: u64, used: u64, free: u64) -> u64 {
    let room = (used + free).saturating_sub(FLOOR);
    if cap < room { cap } else { room }
}

/// Whether the system is short of memory: macOS's warning level or worse.
/// Always false elsewhere.
pub fn memory_pressure() -> bool {
    #[cfg(target_os = "macos")]
    {
        let mut level: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        // SAFETY: the name is NUL-terminated and `level` is a live int of `size` bytes.
        let read = unsafe {
            libc::sysctlbyname(
                c"kern.memorystatus_vm_pressure_level".as_ptr(),
                (&raw mut level).cast(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        // 1 normal, 2 warning, 4 critical.
        read == 0 && level >= 2
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Peak memory of a separation: a fixed part, and a part per minute of audio.
const SEPARATION_BASE: u64 = 1_800_000_000;
const SEPARATION_PER_MINUTE: u64 = 155_000_000;

/// Longest track, in minutes, whose separation fits in three quarters of this
/// machine's memory. Assumes 8 GB where memory is not read.
pub fn longest_track_minutes() -> u32 {
    static LONGEST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LONGEST.get_or_init(|| longest_for(physical_memory().unwrap_or(8 << 30)))
}

fn longest_for(ram: u64) -> u32 {
    let room = (ram / 4 * 3).saturating_sub(SEPARATION_BASE);
    u32::try_from(room / SEPARATION_PER_MINUTE)
        .unwrap_or(u32::MAX)
        .max(1)
}

/// Bytes of physical memory, or `None` where it is not read.
pub fn physical_memory() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf only reads system constants.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        (pages > 0 && size > 0).then(|| pages as u64 * size as u64)
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..MEMORYSTATUSEX::default()
        };
        // SAFETY: `status` is a live MEMORYSTATUSEX with its length set.
        (unsafe { GlobalMemoryStatusEx(&raw mut status) } != 0).then_some(status.ullTotalPhys)
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

#[cfg(unix)]
fn free_at(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: an all-zero statvfs is a valid value for the call to overwrite.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `name` is NUL-terminated and `stat` is a live statvfs.
    if unsafe { libc::statvfs(name.as_ptr(), &raw mut stat) } != 0 {
        return None;
    }
    #[allow(clippy::useless_conversion)]
    Some(u64::from(stat.f_bavail) * u64::from(stat.f_frsize))
}

#[cfg(windows)]
fn free_at(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut free = 0u64;
    // SAFETY: `wide` is NUL-terminated and `free` is a live u64; the other two
    // outputs are optional.
    let ok = unsafe {
        windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &raw mut free,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (ok != 0).then_some(free)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_longest_track_follows_the_machines_memory() {
        assert_eq!(longest_for(8 << 30), 29);
        assert_eq!(longest_for(16 << 30), 71);
        assert_eq!(longest_for(64 << 30), 320);
        assert_eq!(longest_for(2 << 30), 1, "never below a minute");
    }

    #[test]
    fn the_temp_volume_has_a_free_size() {
        let free = free_bytes(&std::env::temp_dir()).expect("statvfs on the temp dir");
        assert!(free > 0);
    }

    #[test]
    fn a_missing_path_is_measured_where_it_would_go() {
        let missing = std::env::temp_dir().join("stemd-no-such-dir/below/it");
        assert!(free_bytes(&missing).is_some());
    }

    #[test]
    fn a_refusal_says_what_is_free_and_what_is_needed() {
        let full = DiskFull {
            free: 1_200_000_000,
            needed: 2_300_000_000,
        };
        assert_eq!(
            full.to_string(),
            "not enough disk space: 1.2 GB free, 2.3 GB needed; free some space and try again"
        );
    }

    #[test]
    fn the_budget_leaves_the_floor_free() {
        let gb = 1_000_000_000;
        // Plenty free: the cap stands.
        assert_eq!(fit(4 * gb, gb, 50 * gb), 4 * gb);
        // 3 GB free with 1 GB held: 2 GB may be held in all.
        assert_eq!(fit(4 * gb, gb, 3 * gb), 2 * gb);
        // Under the floor already: nothing.
        assert_eq!(fit(4 * gb, 0, gb), 0);
    }
}
