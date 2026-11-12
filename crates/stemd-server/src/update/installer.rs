//! Windows: download Setup, check it against the digest GitHub recorded, and
//! start it so it can replace this copy once the app has quit.
//!
//! Setup runs silently with this copy's install mode, and with `/relaunch`
//! when the app closes for it: it reuses the location and GPU choice of the
//! copy it replaces, and starts Stemd again when it is done.

use std::io::Read;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::github::Release;

/// Setup, downloaded and verified.
#[derive(Clone, Debug)]
pub struct Setup {
    path: PathBuf,
    sha256: String,
}

/// Where downloads go.
fn download_dir() -> PathBuf {
    std::env::temp_dir().join("stemd-update")
}

/// Remove earlier downloads. Setup started by the previous run may still hold
/// its file, which is then left for the next launch.
pub fn remove_downloads() {
    let _ = std::fs::remove_dir_all(download_dir());
}

/// An install mode, as Setup's command line names it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    AllUsers,
    CurrentUser,
}

impl Mode {
    fn arg(self) -> &'static str {
        match self {
            Self::AllUsers => "/AllUsers",
            Self::CurrentUser => "/CurrentUser",
        }
    }
}

/// How Setup installed this copy, from the uninstall key naming its folder; `None` for the zip.
pub fn install_mode() -> Option<Mode> {
    use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};

    let here = std::env::current_exe().ok()?.parent()?.to_path_buf();
    [
        (HKEY_CURRENT_USER, Mode::CurrentUser),
        (HKEY_LOCAL_MACHINE, Mode::AllUsers),
    ]
    .into_iter()
    .find(|(hive, _)| install_location(*hive).is_some_and(|there| same_dir(&here, &there)))
    .map(|(_, mode)| mode)
}

/// Whether this account can install in `mode`: all users needs an administrator account.
pub fn can_install(mode: Mode) -> bool {
    mode == Mode::CurrentUser || can_elevate()
}

/// Whether this account is an administrator, elevated or not.
fn can_elevate() -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{
        GetTokenInformation, TOKEN_ELEVATION_TYPE, TOKEN_QUERY, TokenElevationType,
        TokenElevationTypeFull, TokenElevationTypeLimited,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
    use windows_sys::Win32::UI::Shell::IsUserAnAdmin;

    let mut token: HANDLE = std::ptr::null_mut();
    let mut kind: TOKEN_ELEVATION_TYPE = 0;
    let mut len = 0u32;
    // SAFETY: the token is this process's, queried into a value of the size
    // given, and closed once read.
    let split = unsafe {
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return false;
        }
        let ok = GetTokenInformation(
            token,
            TokenElevationType,
            (&mut kind as *mut TOKEN_ELEVATION_TYPE).cast(),
            std::mem::size_of::<TOKEN_ELEVATION_TYPE>() as u32,
            &mut len,
        );
        CloseHandle(token);
        ok != 0 && (kind == TokenElevationTypeFull || kind == TokenElevationTypeLimited)
    };
    // Without a split token (UAC off, or a standard user), check the group.
    // SAFETY: no arguments.
    split || unsafe { IsUserAnAdmin() != 0 }
}

/// Download `release`'s Setup and check its size and SHA-256. `progress`
/// receives the fraction done.
pub fn download(release: &Release, mut progress: impl FnMut(f32)) -> Result<Setup> {
    let asset = release
        .installer()
        .context("the release has no Setup for Windows")?;
    let expected = asset
        .sha256()
        .context("the release lists no SHA-256 for Setup")?;
    let dir = download_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    crate::disk::ensure_room_for(&dir, asset.size)?;

    // No timeout covers a single read: `release` abandons a download with no
    // progress for 60 s.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(Duration::from_secs(60 * 60)))
        .build()
        .into();
    let response = agent
        .get(&asset.browser_download_url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {}: {e}", asset.browser_download_url))?;
    let partial = dir.join(format!("{}.part", asset.name));
    let total = asset.size.max(1) as f32;
    let streamed =
        crate::models::stream_to_file(response.into_body().into_reader(), &partial, |done| {
            progress(done as f32 / total)
        });
    let (digest, written) = match streamed {
        Ok(done) => done,
        Err(err) => {
            let _ = std::fs::remove_file(&partial);
            return Err(err);
        }
    };
    if written != asset.size || digest != expected {
        let _ = std::fs::remove_file(&partial);
        bail!(
            "{} does not match the release: {written} bytes with SHA-256 {digest}, \
             expected {} bytes with {expected}",
            asset.name,
            asset.size
        );
    }
    progress(1.0);
    let path = dir.join(&asset.name);
    std::fs::rename(&partial, &path).with_context(|| format!("publishing {}", path.display()))?;
    tracing::info!("downloaded {}", path.display());
    Ok(Setup {
        path,
        sha256: expected,
    })
}

impl Setup {
    /// Check the file again and start it silently in `mode`, with `/relaunch` to restart Stemd.
    pub fn run(&self, mode: Mode, relaunch: bool) -> Result<()> {
        // Read sharing only, so the file cannot change before Setup starts.
        const FILE_SHARE_READ: u32 = 1;
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0; 1 << 20];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let digest = format!("{:x}", hasher.finalize());
        if digest != self.sha256 {
            bail!("{} changed after it was verified", self.path.display());
        }
        tracing::info!("starting {}", self.path.display());
        let args = format!(
            "/S {}{}",
            mode.arg(),
            if relaunch { " /relaunch" } else { "" }
        );
        shell_open(&self.path, Some(&args)).map_err(anyhow::Error::msg)?;
        drop(file);
        Ok(())
    }
}

/// Whether two directories are the same: resolved as Windows resolves them
/// (case, short names, links), then compared without a trailing separator.
fn same_dir(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .trim_end_matches(['\\', '/'])
            .to_lowercase()
    };
    norm(a) == norm(b)
}

/// `InstallLocation` from stemd's uninstall key in `hive`, in the 64-bit
/// registry view.
fn install_location(hive: windows_sys::Win32::System::Registry::HKEY) -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Registry::{
        RRF_RT_REG_SZ, RRF_SUBKEY_WOW6464KEY, RegGetValueW,
    };

    let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
    let key = wide(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\stemd");
    let value = wide("InstallLocation");
    let read = |buf: *mut u16, bytes: &mut u32| {
        // SAFETY: NUL-terminated names, and `bytes` is the size of `buf`, or
        // 0 with a null `buf` to ask for the size.
        unsafe {
            RegGetValueW(
                hive,
                key.as_ptr(),
                value.as_ptr(),
                RRF_RT_REG_SZ | RRF_SUBKEY_WOW6464KEY,
                std::ptr::null_mut(),
                buf.cast(),
                bytes,
            )
        }
    };
    let mut bytes = 0u32;
    if read(std::ptr::null_mut(), &mut bytes) != 0 {
        return None;
    }
    let mut buf = vec![0u16; (bytes as usize).div_ceil(2)];
    if read(buf.as_mut_ptr(), &mut bytes) != 0 {
        return None;
    }
    // `bytes` includes the terminating NUL.
    buf.truncate((bytes as usize / 2).saturating_sub(1));
    Some(PathBuf::from(std::ffi::OsString::from_wide(&buf)))
}

/// Open a file or URL as Explorer would, with `args` for a program. Setup asks
/// for elevation in its manifest, which only the shell honours:
/// `CreateProcess` refuses it.
pub fn shell_open(target: &Path, args: Option<&str>) -> Result<(), String> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let wide = |s: &std::ffi::OsStr| s.encode_wide().chain([0]).collect::<Vec<u16>>();
    let verb = wide("open".as_ref());
    let file = wide(target.as_os_str());
    let args = args.map(|a| wide(a.as_ref()));
    // SAFETY: every string is NUL-terminated and outlives the call.
    let code = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            file.as_ptr(),
            args.as_ref().map_or(std::ptr::null(), |a| a.as_ptr()),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    } as isize;
    // Values up to 32 are error codes.
    if code > 32 {
        Ok(())
    } else {
        Err(format!("ShellExecute failed with code {code}"))
    }
}
