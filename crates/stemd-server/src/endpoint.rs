//! The address the API listens on. The window can move it to another port while
//! the server runs.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::api::AppState;

/// Listeners bound for a new port, for the serve loop to take over.
pub type Moves = UnboundedReceiver<std::net::TcpListener>;

pub struct Endpoint {
    ip: std::net::IpAddr,
    port: AtomicU16,
    /// The saved port another app held at launch, 0 for none. Cleared by a move.
    taken: AtomicU16,
    moves: UnboundedSender<std::net::TcpListener>,
}

/// Why the server could not move to a port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveError {
    InUse(u16),
    Failed(String),
}

impl std::fmt::Display for MoveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InUse(port) if cfg!(target_os = "macos") && matches!(port, 5000 | 7000) => {
                write!(f, "Another app is using {port} (AirPlay Receiver).")
            }
            Self::InUse(port) => write!(f, "Another app is using {port}."),
            Self::Failed(reason) => write!(f, "Could not listen there: {reason}."),
        }
    }
}

impl Endpoint {
    /// The endpoint for a listener already bound at `bound`, and the receiving
    /// end of its moves. `taken` is the saved port that could not be bound.
    pub fn new(bound: SocketAddr, taken: Option<u16>) -> (Self, Moves) {
        let (moves, receiver) = unbounded_channel();
        let endpoint = Self {
            ip: bound.ip(),
            port: AtomicU16::new(bound.port()),
            taken: AtomicU16::new(taken.unwrap_or(0)),
            moves,
        };
        (endpoint, receiver)
    }

    pub fn port(&self) -> u16 {
        self.port.load(Ordering::SeqCst)
    }

    /// The saved port another app held at launch, until the server moves.
    pub fn taken(&self) -> Option<u16> {
        Some(self.taken.load(Ordering::SeqCst)).filter(|&port| port != 0)
    }

    pub fn addr(&self) -> SocketAddr {
        SocketAddr::new(self.ip, self.port())
    }

    /// Bind `port` on the same interface and hand it to the serve loop, which
    /// stops accepting on the old port and lets its open requests finish.
    pub(crate) fn move_to(&self, port: u16) -> Result<(), MoveError> {
        let addr = SocketAddr::new(self.ip, port);
        let listener = crate::serve::bind(addr).map_err(|err| {
            match err
                .downcast_ref::<std::io::Error>()
                .map(std::io::Error::kind)
            {
                Some(std::io::ErrorKind::AddrInUse) => MoveError::InUse(port),
                _ => MoveError::Failed(format!("{:#}", err.root_cause())),
            }
        })?;
        self.moves
            .send(listener)
            .map_err(|_| MoveError::Failed("the server is not running".into()))?;
        self.port.store(port, Ordering::SeqCst);
        self.taken.store(0, Ordering::SeqCst);
        Ok(())
    }
}

impl AppState {
    /// Serve on `port` from now on: rebind, advertise it, and save it.
    pub fn move_to_port(&self, port: u16) -> Result<(), MoveError> {
        if port == self.endpoint.port() {
            return Ok(());
        }
        self.endpoint.move_to(port)?;
        self.serving.move_to(port, &self.queue.info().model);
        self.settings.set_port(port);
        tracing::info!("listening on http://{}", self.endpoint.addr());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_move_hands_the_new_listener_over_and_a_taken_port_is_refused() {
        let first = crate::serve::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let (endpoint, mut moves) = Endpoint::new(first.local_addr().unwrap(), Some(1));
        assert_eq!(endpoint.taken(), Some(1));

        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        endpoint.move_to(port).expect("a free port");
        assert_eq!(endpoint.port(), port);
        assert_eq!(endpoint.taken(), None, "a move clears the launch fallback");
        let moved = moves.try_recv().expect("the listener was handed over");
        assert_eq!(moved.local_addr().unwrap().port(), port);

        let taken = first.local_addr().unwrap().port();
        assert_eq!(endpoint.move_to(taken), Err(MoveError::InUse(taken)));
        assert_eq!(endpoint.port(), port, "a refused move keeps the port");
    }
}
