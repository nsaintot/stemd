//! Finding a stemd server over mDNS.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

const SERVICE_TYPE: &str = "_stemd._tcp.local.";

/// How long to block on each receive before re-checking the deadline.
const TICK: Duration = Duration::from_millis(250);

/// Browse for a server, returning the first one that resolves to an IPv4
/// address as `host:port`.
pub fn discover(timeout: Duration) -> Result<String> {
    let daemon = mdns_sd::ServiceDaemon::new().context("starting mDNS browser")?;
    let receiver = daemon.browse(SERVICE_TYPE).context("browsing")?;
    let deadline = Instant::now() + timeout;

    while Instant::now() < deadline {
        if let Ok(mdns_sd::ServiceEvent::ServiceResolved(info)) = receiver.recv_timeout(TICK)
            && let Some(addr) = info.get_addresses().iter().find(|a| a.is_ipv4())
        {
            let host = format!("{addr}:{}", info.get_port());
            println!("discovered   : {} at {host}", info.get_fullname());
            let _ = daemon.shutdown();
            return Ok(host);
        }
    }

    let _ = daemon.shutdown();
    bail!(
        "no stemd server found on the network after {}s — is it running? \
         start it with `open dist/stemd.app`, or pass --host host:port",
        timeout.as_secs()
    )
}
