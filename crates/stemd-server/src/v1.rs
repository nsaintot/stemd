//! Protocol 1, which is deprecated and may stop working in a future release.
//!
//! Every answer under `/v1` carries `Deprecation: true` and a `Link` to the
//! API reference. The first request from each client address logs a warning.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::LazyLock;

use axum::extract::{ConnectInfo, Request};
use axum::http::{HeaderValue, header};
use axum::middleware::Next;
use axum::response::Response;
use parking_lot::Mutex;

/// The `Link` header on every protocol 1 answer.
const LINK: &str = "<https://github.com/nsaintot/stemd/blob/main/docs/api.md>; rel=\"deprecation\"; type=\"text/html\"";

/// True for a protocol 1 path.
fn is_v1(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

/// Mark a protocol 1 answer deprecated, and warn once per client address.
pub async fn deprecate(request: Request, next: Next) -> Response {
    if !is_v1(request.uri().path()) {
        return next.run(request).await;
    }
    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>()
        && first_from(addr.ip())
    {
        tracing::warn!(
            "{} uses protocol 1 (/v1), which is deprecated and may stop working in a future release",
            addr.ip()
        );
    }
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert("deprecation", HeaderValue::from_static("true"));
    headers.insert(header::LINK, HeaderValue::from_static(LINK));
    response
}

/// True the first time `ip` is seen.
fn first_from(ip: IpAddr) -> bool {
    static SEEN: LazyLock<Mutex<Seen>> = LazyLock::new(Mutex::default);
    SEEN.lock().first(ip)
}

/// The most client addresses remembered; past it they are all forgotten.
const SEEN_MAX: usize = 1024;

/// The client addresses already warned about.
#[derive(Default)]
struct Seen(HashSet<IpAddr>);

impl Seen {
    /// True the first time `ip` is seen.
    fn first(&mut self, ip: IpAddr) -> bool {
        if self.0.len() >= SEEN_MAX && !self.0.contains(&ip) {
            self.0.clear();
        }
        self.0.insert(ip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::routing::get;
    use tower::ServiceExt as _;

    async fn get_from(app: &Router, path: &str) -> Response {
        let mut request = Request::builder().uri(path).body(Body::empty()).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from(([192, 0, 2, 7], 50_000))));
        app.clone().oneshot(request).await.unwrap()
    }

    #[tokio::test]
    async fn a_v1_answer_is_marked_deprecated() {
        let app = Router::new()
            .route("/v1/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(deprecate));
        for path in ["/v1/health", "/v1/nothing"] {
            let response = get_from(&app, path).await;
            assert_eq!(response.headers()["deprecation"], "true", "{path}");
            assert!(
                response.headers()[header::LINK]
                    .to_str()
                    .unwrap()
                    .contains("rel=\"deprecation\""),
                "{path}"
            );
        }
        let response = get_from(&app, "/v10/health").await;
        assert!(response.headers().get("deprecation").is_none());
    }

    #[test]
    fn a_client_address_is_warned_about_once() {
        let ip = IpAddr::from([198, 51, 100, 1]);
        assert!(first_from(ip));
        assert!(!first_from(ip));
        assert!(first_from(IpAddr::from([198, 51, 100, 2])));
    }

    #[test]
    fn the_warned_addresses_are_forgotten_past_the_cap() {
        let mut seen = Seen::default();
        for n in 0..SEEN_MAX as u32 {
            assert!(seen.first(IpAddr::from(n.to_be_bytes())));
        }
        assert!(!seen.first(IpAddr::from(0u32.to_be_bytes())));
        assert!(seen.first(IpAddr::from([203, 0, 113, 1])));
        assert_eq!(seen.0.len(), 1);
    }
}
