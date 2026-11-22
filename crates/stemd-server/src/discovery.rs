//! mDNS advertisement through the responder the system already runs.
//!
//! A client finds the host by browsing `_stemd._tcp`. The server registers with the system
//! responder, which keeps answering and handles interface changes, sleep and wake, and name
//! conflicts, so stemd runs no responder of its own.
//!
//! [`backend`] is chosen at compile time:
//!
//! ```text
//! macOS    mDNSResponder, DNSServiceRegister out of libSystem
//! Linux    avahi, the same symbols out of avahi-compat's libdns_sd
//! Windows  the DNS Client service, DnsServiceRegister out of dnsapi.dll
//! ```
//!
//! The first two are one API; Windows has its own module below.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};

pub const SERVICE_TYPE: &str = "_stemd._tcp";

/// Holds the registration. Call [`Advertiser::withdraw`] before exiting.
///
/// `Drop` alone is not enough: a macOS Cmd-Q terminates without unwinding, a
/// signal does not unwind, and [`crate::shutdown::now`] leaves through `_exit`, so
/// destructors never run. `Drop` is the backstop.
pub struct Advertiser {
    registration: backend::Registration,
    /// The fully qualified name the system granted on the network, which can differ from the name
    /// asked for. See [`Advertiser::start`].
    name: String,
    withdrawn: AtomicBool,
}

impl Advertiser {
    /// Announce this server over mDNS with TXT keys `v`, `version`, `model`, `path`, `features`.
    pub fn start(instance: &str, port: u16, model: &str) -> Result<Self> {
        let mut props = HashMap::new();
        props.insert("v", "2".to_owned());
        props.insert("version", env!("CARGO_PKG_VERSION").to_owned());
        props.insert("model", model.to_owned());
        props.insert("path", "/v2".to_owned());
        props.insert("features", crate::api::health::FEATURES.join(","));

        // Log the interfaces at debug, so a discovery failure on a multi-homed host can be
        // diagnosed from one line; the log buffer keeps debug lines.
        let addrs = local_addrs();
        if addrs.is_empty() {
            bail!("no non-loopback IPv4 interface to advertise on");
        }
        tracing::debug!(
            "interfaces: {}",
            addrs
                .iter()
                .map(|(n, ip)| format!("{n}={ip}"))
                .collect::<Vec<_>>()
                .join(" ")
        );

        let registration = backend::register(instance, port, &props, &addrs)?;
        // Log the granted name, which mDNS changes on a collision, such as `stemd(5)` for `stemd`.
        // Clients search for the granted name.
        let name = registration
            .granted_name()
            .unwrap_or_else(|| full_name(instance));
        tracing::info!("advertising {name} on port {port}");
        Ok(Self {
            registration,
            name,
            withdrawn: AtomicBool::new(false),
        })
    }

    /// Withdraw the registration. Idempotent.
    ///
    /// The responder sends the goodbye packet. The backend returns a [`Withdrawal`], which is
    /// logged.
    pub fn withdraw(&self) {
        if self.withdrawn.swap(true, Ordering::SeqCst) {
            return;
        }
        let full = &self.name;
        match self.registration.withdraw() {
            Withdrawal::Done => tracing::info!("withdrew {full}"),
            Withdrawal::Refused => tracing::warn!(
                "could not withdraw {full}, so clients will keep offering it \
                 until the record expires"
            ),
            Withdrawal::Pending => {
                tracing::info!("asked the system to withdraw {full}; not waiting")
            }
        }
    }
}

impl Drop for Advertiser {
    fn drop(&mut self) {
        self.withdraw();
    }
}

/// What is known about a withdrawal's outcome.
///
/// Windows hands the work to the DNS Client service and does not wait for the answer, so the
/// outcome on exit can be unknown. On the Bonjour platforms, only `Done` occurs.
#[cfg_attr(not(windows), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Withdrawal {
    /// The system confirmed it.
    Done,
    /// The system refused it, and the reason is in the log above.
    Refused,
    /// Handed to the system without waiting for the result.
    Pending,
}

/// The fully qualified instance name, `<instance>._stemd._tcp.local`.
///
/// Bonjour builds this from the three parts it is handed separately; Windows
/// wants it already assembled.
#[cfg_attr(not(windows), allow(dead_code))]
fn full_name(instance: &str) -> String {
    format!("{instance}.{SERVICE_TYPE}.local")
}

/// True when two instance names refer to the same service.
///
/// On Windows, this tells a granted name apart from a rename after a collision. DNS names are
/// case-insensitive and the root dot is optional, so neither difference counts as a rename.
#[cfg_attr(not(windows), allow(dead_code))]
fn same_name(a: &str, b: &str) -> bool {
    a.trim_end_matches('.')
        .eq_ignore_ascii_case(b.trim_end_matches('.'))
}

/// Non-loopback IPv4 interfaces, as `(name, address)`.
///
/// Diagnostics on macOS and Linux: Bonjour picks the interfaces itself. On
/// Windows the first of these also becomes the advertised A record; see
/// [`backend::register`] there.
fn local_addrs() -> Vec<(String, IpAddr)> {
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .filter(|i| !i.is_loopback() && i.addr.ip().is_ipv4())
        .map(|i| (i.name, i.addr.ip()))
        .collect()
}

/// The first advertisable IPv4 address, if there is one.
#[cfg_attr(not(windows), allow(dead_code))]
fn first_ipv4(addrs: &[(String, IpAddr)]) -> Option<Ipv4Addr> {
    addrs.iter().find_map(|(_, ip)| match ip {
        IpAddr::V4(v4) => Some(*v4),
        IpAddr::V6(_) => None,
    })
}

/// Encode TXT entries for DNS-SD: each entry is a length byte followed by `key=value`.
///
/// Bonjour takes this wire form; Windows takes the pairs unencoded and encodes them itself.
#[cfg_attr(windows, allow(dead_code))]
fn encode_txt(props: &HashMap<&str, String>) -> Vec<u8> {
    let mut out = Vec::new();
    // Sorted so the record is byte-stable across runs.
    let mut keys: Vec<_> = props.keys().collect();
    keys.sort_unstable();
    for key in keys {
        let entry = format!("{key}={}", props[key]);
        // A single entry cannot exceed 255 bytes. Ours are far shorter; an entry that is too long
        // is skipped with a warning.
        match u8::try_from(entry.len()) {
            Ok(len) => {
                out.push(len);
                out.extend_from_slice(entry.as_bytes());
            }
            Err(_) => tracing::warn!("TXT entry {key} too long, omitted"),
        }
    }
    out
}

/// Bonjour, on macOS and on Linux through avahi-compat.
// `DNSServiceRegister` is in libSystem on macOS. On Linux it comes from `libdns_sd` (package
// `libavahi-compat-libdnssd-dev`), a shim over avahi that runs no responder of its own and
// implements both calls used here. The registration lasts as long as the `DNSServiceRef`.
#[cfg(unix)]
mod backend {
    use std::collections::HashMap;
    use std::ffi::{CString, c_char, c_void};
    use std::net::IpAddr;

    use anyhow::{Context, Result, bail};

    type DnsServiceRef = *mut c_void;
    type DnsServiceErrorType = i32;

    // Linked implicitly out of libSystem on macOS; named explicitly on Linux,
    // where it is a package that has to be installed.
    #[cfg_attr(target_os = "linux", link(name = "dns_sd"))]
    unsafe extern "C" {
        fn DNSServiceRegister(
            sd_ref: *mut DnsServiceRef,
            flags: u32,
            interface_index: u32,
            name: *const c_char,
            regtype: *const c_char,
            domain: *const c_char,
            host: *const c_char,
            port: u16,
            txt_len: u16,
            txt_record: *const c_void,
            callback: *const c_void,
            context: *mut c_void,
        ) -> DnsServiceErrorType;

        fn DNSServiceRefDeallocate(sd_ref: DnsServiceRef);
    }

    pub struct Registration {
        sd_ref: DnsServiceRef,
    }

    // The ref is touched at construction and at withdrawal, and `Advertiser`
    // serialises those with an `AtomicBool`. The responder itself is
    // thread-safe.
    unsafe impl Send for Registration {}
    unsafe impl Sync for Registration {}

    pub fn register(
        instance: &str,
        port: u16,
        props: &HashMap<&str, String>,
        _addrs: &[(String, IpAddr)],
    ) -> Result<Registration> {
        let txt = super::encode_txt(props);
        let name = CString::new(instance).context("instance name contains a NUL")?;
        let regtype = CString::new(super::SERVICE_TYPE).expect("static string");

        let mut sd_ref: DnsServiceRef = std::ptr::null_mut();
        // SAFETY: all pointers outlive the call; `port` is network byte order
        // as the API requires; a null callback means we do not want completion
        // notifications, which also means no run loop to pump.
        let err = unsafe {
            DNSServiceRegister(
                &mut sd_ref,
                0,
                0, // all interfaces
                name.as_ptr(),
                regtype.as_ptr(),
                std::ptr::null(), // default domain (.local)
                std::ptr::null(), // default host
                port.to_be(),
                u16::try_from(txt.len()).context("TXT record too large")?,
                txt.as_ptr().cast(),
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };
        if err != 0 {
            bail!("DNSServiceRegister failed with {err}");
        }
        Ok(Registration { sd_ref })
    }

    impl Registration {
        /// Always `None`: Bonjour reports the granted name to the registration callback, and this
        /// passes a null callback to avoid pumping a run loop for the life of the process. The
        /// caller falls back to the name it asked for.
        pub fn granted_name(&self) -> Option<String> {
            None
        }

        /// Deallocating the ref closes the connection to the responder, which sends the
        /// goodbye packet.
        ///
        /// Always [`super::Withdrawal::Done`]: the call returns nothing and cannot fail. The
        /// three-way answer is for Windows.
        pub fn withdraw(&self) -> super::Withdrawal {
            // SAFETY: called at most once, guarded by `Advertiser::withdrawn`.
            unsafe { DNSServiceRefDeallocate(self.sd_ref) };
            super::Withdrawal::Done
        }
    }
}

/// Windows, through the DNS Client service.
///
/// This API differs from Bonjour in three ways, so it has its own module:
///
/// - It is asynchronous: `DnsServiceRegister` returns `DNS_REQUEST_PENDING`, and the outcome
///   arrives on a callback that the system runs on its own thread.
/// - It takes the service instance as an object built by `DnsServiceConstructInstance`, with TXT
///   pairs unencoded.
/// - The request structure owns the registration, so the structure must outlive it and is boxed.
#[cfg(windows)]
mod backend {
    use std::collections::HashMap;
    use std::ffi::{OsStr, c_void};
    use std::net::IpAddr;
    use std::os::windows::ffi::OsStrExt;
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use anyhow::{Result, bail};

    /// `DNS_REQUEST_PENDING`, the success return from an async DNS call.
    const DNS_REQUEST_PENDING: i32 = 9506;
    /// `DNS_QUERY_REQUEST_VERSION1`, the only version these structures have.
    const REQUEST_VERSION1: u32 = 1;

    /// How long to wait for the system to confirm a registration.
    ///
    /// The call is local IPC to a service on the same machine, so this only bounds a fast path. It
    /// runs once, at startup.
    const COMPLETION_TIMEOUT: Duration = Duration::from_secs(3);

    /// How long to wait for a withdrawal on the way out.
    ///
    /// The DNS Client service does the work in another process, which our exit does not interrupt,
    /// so the wait only makes the log line accurate. `DnsServiceDeRegister` can take seconds and
    /// then report a failure that nothing here can act on.
    const WITHDRAW_TIMEOUT: Duration = Duration::from_millis(250);

    /// `DNS_SERVICE_REGISTER_REQUEST`. It must outlive the registration: the API keeps the pointer,
    /// and `DnsServiceDeRegister` takes this structure again in place of a handle.
    ///
    /// Most fields are written for the ABI and never read back, which `dead_code` flags.
    #[repr(C)]
    #[allow(dead_code)]
    struct RegisterRequest {
        version: u32,
        interface_index: u32,
        service_instance: *mut c_void,
        completion: Option<CompletionFn>,
        query_context: *mut c_void,
        credentials: *mut c_void,
        unicast_enabled: i32,
    }

    /// `DNS_SERVICE_CANCEL`, always passed as null here: a registration is withdrawn once it exists
    /// and is never cancelled while pending. The type is named so the two calls below match their
    /// documented signatures.
    #[repr(C)]
    #[allow(dead_code)]
    struct ServiceCancel {
        reserved: *mut c_void,
    }

    type CompletionFn =
        unsafe extern "system" fn(status: u32, context: *mut c_void, instance: *mut c_void);

    // `DNS_SERVICE_INSTANCE` stays opaque: the library constructs and frees it, and nothing here
    // reads its fields except the first, so declaring its layout would only risk an error.
    #[link(name = "dnsapi")]
    unsafe extern "system" {
        fn DnsServiceConstructInstance(
            service_name: *const u16,
            host_name: *const u16,
            ip4: *const u32,
            ip6: *const u8,
            port: u16,
            priority: u16,
            weight: u16,
            properties_count: u32,
            keys: *const *const u16,
            values: *const *const u16,
        ) -> *mut c_void;

        fn DnsServiceFreeInstance(instance: *mut c_void);

        fn DnsServiceRegister(request: *mut RegisterRequest, cancel: *mut ServiceCancel) -> i32;

        fn DnsServiceDeRegister(request: *mut RegisterRequest, cancel: *mut ServiceCancel) -> i32;
    }

    /// What the system reported.
    ///
    /// The instance is stored as an integer, since it crosses from a system thread and a raw
    /// pointer is not `Send`. It is zero when the API passed no instance.
    #[derive(Clone, Copy)]
    struct Outcome {
        status: u32,
        instance: usize,
    }

    /// A rendezvous with the system's completion thread.
    #[derive(Default)]
    struct Completion {
        outcome: Mutex<Option<Outcome>>,
        ready: Condvar,
    }

    impl Completion {
        /// What was reported, or `None` if nothing was within `patience`.
        fn wait(&self, patience: Duration) -> Option<Outcome> {
            let guard = self.outcome.lock().ok()?;
            let (guard, timeout) = self
                .ready
                .wait_timeout_while(guard, patience, |o| o.is_none())
                .ok()?;
            if timeout.timed_out() { None } else { *guard }
        }
    }

    /// The instance name inside a `DNS_SERVICE_INSTANCE`.
    ///
    /// `pszInstanceName` is that structure's first member, so this reads offset zero
    /// and leaves the rest of the layout opaque.
    ///
    /// # Safety
    ///
    /// `instance` must be null, or a `DNS_SERVICE_INSTANCE` from the API.
    unsafe fn instance_name(instance: *mut c_void) -> Option<String> {
        if instance.is_null() {
            return None;
        }
        // SAFETY: the caller guarantees the pointee, and its first member is a
        // NUL-terminated wide string.
        let name = unsafe { *instance.cast::<*const u16>() };
        if name.is_null() {
            return None;
        }
        let mut len = 0;
        // SAFETY: as above. The string is NUL-terminated.
        while unsafe { *name.add(len) } != 0 {
            len += 1;
        }
        // SAFETY: `len` is the length up to the NUL, so this stays inside it.
        let chars = unsafe { std::slice::from_raw_parts(name, len) };
        Some(String::from_utf16_lossy(chars))
    }

    /// The completion callback for both register and deregister.
    // It runs on a system thread: it records the status, wakes the caller and returns. Nothing in
    // it may panic, because unwinding out of an `extern "system"` frame aborts the process.
    //
    // The instance it receives describes the registered service, which after a collision has a
    // different name, so it is passed to the caller. The caller does not free it, since there is
    // one per process.
    unsafe extern "system" fn on_complete(
        status: u32,
        context: *mut c_void,
        instance: *mut c_void,
    ) {
        if context.is_null() {
            return;
        }
        // SAFETY: the context is an `Arc<Completion>` leaked by the caller
        // exactly once per request, and this callback fires exactly once per
        // request, so this reclaims that reference and no other.
        let completion = unsafe { Arc::from_raw(context.cast::<Completion>()) };
        if let Ok(mut slot) = completion.outcome.lock() {
            *slot = Some(Outcome {
                status,
                instance: instance as usize,
            });
        }
        completion.ready.notify_all();
    }

    /// A NUL-terminated UTF-16 string, kept alive by the caller.
    fn wide(s: &str) -> Vec<u16> {
        OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    pub struct Registration {
        request: Box<RegisterRequest>,
        /// The instance handed to `DnsServiceRegister`, kept because the request
        /// still points at it.
        constructed: *mut c_void,
        /// The instance the system reported registering, which is the one a
        /// withdrawal has to name. See [`register`].
        registered: *mut c_void,
    }

    // Touched at construction and at withdrawal, which `Advertiser` serialises
    // with an `AtomicBool`. The DNS Client service is itself thread-safe.
    unsafe impl Send for Registration {}
    unsafe impl Sync for Registration {}

    pub fn register(
        instance: &str,
        port: u16,
        props: &HashMap<&str, String>,
        addrs: &[(String, IpAddr)],
    ) -> Result<Registration> {
        // Sorted, as in `encode_txt`, so the record is byte-stable across runs.
        let mut keys: Vec<&&str> = props.keys().collect();
        keys.sort_unstable();
        let key_bufs: Vec<Vec<u16>> = keys.iter().map(|k| wide(k)).collect();
        let value_bufs: Vec<Vec<u16>> = keys.iter().map(|k| wide(&props[**k])).collect();
        let key_ptrs: Vec<*const u16> = key_bufs.iter().map(|b| b.as_ptr()).collect();
        let value_ptrs: Vec<*const u16> = value_bufs.iter().map(|b| b.as_ptr()).collect();

        let name = wide(&super::full_name(instance));

        // Bonjour defaults the host to the machine's own name. Here it must be given, as a `.local`
        // name, because the A record below is published under that name.
        let host = wide(&format!("{}.local", hostname()));

        // The address itself. `IP4_ADDRESS` is an `in_addr`, in network byte order, which is what
        // `from_ne_bytes` of the octets leaves in memory.
        //
        // This publishes one A record, on whichever interface `if_addrs` lists first; on a host
        // with a Hyper-V switch, a WSL adapter or a Docker bridge, that address may be unreachable
        // from the real network.
        let chosen = super::first_ipv4(addrs);
        if let Some(v4) = chosen {
            tracing::info!(
                "publishing {v4} as the address of {}, chosen from {} interface(s)",
                super::full_name(instance),
                addrs.len()
            );
        }
        let ip4 = chosen.map(|v4| u32::from_ne_bytes(v4.octets()));

        // SAFETY: every pointer below outlives the call, which copies what it
        // needs into the instance it returns.
        let service = unsafe {
            DnsServiceConstructInstance(
                name.as_ptr(),
                host.as_ptr(),
                ip4.as_ref().map_or(std::ptr::null(), |a| a as *const u32),
                std::ptr::null(),
                port,
                0, // priority
                0, // weight
                u32::try_from(key_ptrs.len()).expect("four properties"),
                key_ptrs.as_ptr(),
                value_ptrs.as_ptr(),
            )
        };
        if service.is_null() {
            bail!("DnsServiceConstructInstance returned null");
        }

        let completion = Arc::new(Completion::default());
        let mut request = Box::new(RegisterRequest {
            version: REQUEST_VERSION1,
            interface_index: 0, // all interfaces
            service_instance: service,
            completion: Some(on_complete),
            query_context: Arc::into_raw(Arc::clone(&completion)) as *mut c_void,
            credentials: std::ptr::null_mut(),
            unicast_enabled: 0,
        });

        // SAFETY: `request` is boxed, so its address is stable, and it is kept
        // in the `Registration` returned below for as long as the registration
        // stands.
        let status = unsafe { DnsServiceRegister(request.as_mut(), std::ptr::null_mut()) };
        if status != DNS_REQUEST_PENDING {
            // SAFETY: the callback never ran, so the leaked context is still
            // ours to reclaim, and the instance is still ours to free.
            unsafe {
                drop(Arc::from_raw(request.query_context.cast::<Completion>()));
                DnsServiceFreeInstance(service);
            }
            bail!("DnsServiceRegister failed with {status}");
        }

        match completion.wait(COMPLETION_TIMEOUT) {
            Some(outcome) if outcome.status == 0 => {
                let registered = outcome.instance as *mut c_void;
                let asked = super::full_name(instance);
                // The granted name can differ from the requested one, because mDNS settles a
                // collision by renaming, and the callback's instance is the only place the new name
                // appears. Deregistering by the requested name fails with 9701 and leaves the entry
                // up for the next run to collide with.
                match unsafe { instance_name(registered) } {
                    Some(granted) if !super::same_name(&granted, &asked) => tracing::warn!(
                        "mDNS name: expected {asked}, got {granted}; something already \
                         holds that name, most likely a previous run that did not \
                         withdraw. Stale entries belong to the DNS Client service \
                         and last until it restarts"
                    ),
                    _ => {}
                }
                Ok(Registration {
                    request,
                    constructed: service,
                    registered,
                })
            }
            Some(outcome) => bail!("DnsServiceRegister completed with {}", outcome.status),
            // The request was accepted and the callback may still fire, so
            // neither the request nor the context can be freed. Leak both and
            // report the failure.
            None => {
                let _ = Box::leak(request);
                bail!("DnsServiceRegister did not complete within {COMPLETION_TIMEOUT:?}");
            }
        }
    }

    impl Registration {
        /// The name the system granted, read back off the instance it returned.
        pub fn granted_name(&self) -> Option<String> {
            // SAFETY: `registered` is null or the instance the API handed to the
            // completion callback, which is what this expects.
            unsafe { instance_name(self.registered) }
        }

        /// The instance a withdrawal must name: the one the system reported registering. The
        /// constructed instance is the fallback when a success arrives with no instance attached.
        fn withdrawable(&self) -> *mut c_void {
            if self.registered.is_null() {
                self.constructed
            } else {
                self.registered
            }
        }

        /// Deregistering is asynchronous too, and takes a request structure that describes the
        /// instance, so this builds a second request around the registered instance, with its own
        /// rendezvous, and leaves the registration's request untouched.
        pub fn withdraw(&self) -> super::Withdrawal {
            let completion = Arc::new(Completion::default());
            let request = Box::into_raw(Box::new(RegisterRequest {
                version: REQUEST_VERSION1,
                interface_index: self.request.interface_index,
                service_instance: self.withdrawable(),
                completion: Some(on_complete),
                query_context: Arc::into_raw(Arc::clone(&completion)) as *mut c_void,
                credentials: std::ptr::null_mut(),
                unicast_enabled: 0,
            }));

            // SAFETY: `request` is a live heap allocation, and it is only
            // reclaimed below once the completion has proved the API is done
            // with it.
            let status = unsafe { DnsServiceDeRegister(request, std::ptr::null_mut()) };
            if status != DNS_REQUEST_PENDING {
                tracing::warn!("DnsServiceDeRegister failed with {status}");
                // SAFETY: the callback never ran; both are still ours.
                unsafe {
                    let request = Box::from_raw(request);
                    drop(Arc::from_raw(request.query_context.cast::<Completion>()));
                }
                return super::Withdrawal::Refused;
            }
            match completion.wait(WITHDRAW_TIMEOUT) {
                Some(outcome) if outcome.status == 0 => {
                    // SAFETY: the completion fired, so the API is done with the
                    // request and the context has already been reclaimed by the
                    // callback.
                    drop(unsafe { Box::from_raw(request) });
                    super::Withdrawal::Done
                }
                Some(outcome) => {
                    tracing::warn!("DnsServiceDeRegister completed with {}", outcome.status);
                    super::Withdrawal::Refused
                }
                // The service still holds the request. The request and its context are leaked on
                // purpose: the callback may still fire, and the process has no later point to free
                // them, since it calls `_exit` next.
                None => {
                    let _ = Box::leak(unsafe { Box::from_raw(request) });
                    tracing::debug!(
                        "DnsServiceDeRegister still pending after {WITHDRAW_TIMEOUT:?}; \
                         leaving it with the DNS Client service"
                    );
                    super::Withdrawal::Pending
                }
            }
        }
    }

    /// The machine's name, for the `.local` host record.
    fn hostname() -> String {
        std::env::var("COMPUTERNAME").unwrap_or_else(|_| "localhost".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_txt_entries_are_length_prefixed_and_sorted() {
        let mut props = HashMap::new();
        props.insert("b", "2".to_owned());
        props.insert("a", "1".to_owned());
        let txt = encode_txt(&props);
        assert_eq!(txt, vec![3, b'a', b'=', b'1', 3, b'b', b'=', b'2']);
    }

    #[test]
    fn full_name_is_the_instance_under_the_service_type() {
        assert_eq!(full_name("studio"), "studio._stemd._tcp.local");
    }

    #[test]
    fn a_trailing_dot_or_a_capital_is_not_a_rename() {
        assert!(same_name(
            "stemd._stemd._tcp.local.",
            "stemd._stemd._tcp.local"
        ));
        assert!(same_name(
            "Stemd._stemd._TCP.local",
            "stemd._stemd._tcp.local"
        ));
    }

    /// The rename that the Windows backend warns about: the suffix mDNS adds to settle a collision
    /// must count as a different name.
    #[test]
    fn the_name_mdns_grants_after_a_collision_is_a_rename() {
        assert!(!same_name(
            "stemd(1)._stemd._tcp.local",
            "stemd._stemd._tcp.local"
        ));
    }
}
