//! Keeps the server at full speed while its window is hidden.
//!
//! macOS App Nap and Windows power throttling slow down a process whose window is hidden or
//! minimized: lower CPU priority, delayed timers, slower disk I/O. The process opts out for its whole
//! run. A hidden window still does no work, because eframe skips the UI of a covered or minimized
//! window.

pub use imp::opt_out;

#[cfg(target_os = "macos")]
mod imp {
    use objc2::rc::Retained;
    use objc2::runtime::{NSObjectProtocol, ProtocolObject};
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

    /// Keeps App Nap off until dropped.
    pub struct OptOut(Retained<ProtocolObject<dyn NSObjectProtocol>>);

    /// Turn App Nap off for this process. The Mac can still sleep when idle.
    pub fn opt_out() -> OptOut {
        OptOut(
            NSProcessInfo::processInfo().beginActivityWithOptions_reason(
                NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
                &NSString::from_str("Serving stem separations"),
            ),
        )
    }

    impl Drop for OptOut {
        fn drop(&mut self) {
            // SAFETY: the token came from `beginActivityWithOptions:reason:`.
            unsafe { NSProcessInfo::processInfo().endActivity(&self.0) };
        }
    }
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, PROCESS_POWER_THROTTLING_CURRENT_VERSION,
        PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_STATE,
        ProcessPowerThrottling, SetProcessInformation,
    };

    /// Keeps power throttling off until dropped.
    pub struct OptOut;

    /// Turn power throttling off for this process.
    pub fn opt_out() -> OptOut {
        if let Err(err) = control(PROCESS_POWER_THROTTLING_EXECUTION_SPEED) {
            tracing::warn!("power throttling opt-out: {err}");
        }
        OptOut
    }

    impl Drop for OptOut {
        fn drop(&mut self) {
            let _ = control(0);
        }
    }

    /// Take control of the throttling policies in `mask` and leave them off; 0 hands them back.
    fn control(mask: u32) -> std::io::Result<()> {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: mask,
            StateMask: 0,
        };
        // SAFETY: `state` is a PROCESS_POWER_THROTTLING_STATE that outlives the call, and the size
        // passed is its own.
        let set = unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                std::ptr::from_ref(&state).cast(),
                std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            )
        };
        if set == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
mod imp {
    /// Linux does not throttle a hidden window's process.
    pub struct OptOut;

    /// Nothing to turn off.
    pub fn opt_out() -> OptOut {
        OptOut
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_opt_out_can_be_taken_and_released() {
        drop(super::opt_out());
    }
}
