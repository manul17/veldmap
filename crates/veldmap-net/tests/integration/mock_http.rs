//! Shared loopback HTTP/1.1 mock server helper for `veldmap-net` integration
//! tests.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use veldmap_core::sync::CancelToken;

/// Parsed HTTP/1.1 request received by [`MockHttpServer`].
pub(crate) struct MockHttpRequest {
    pub(crate) path_and_query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl MockHttpRequest {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub(crate) fn body_string(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// Read full HTTP/1.1 request headers and any `Content-Length` body from
/// `stream`, returning `None` on EOF, I/O error, or `stop` cancellation.
pub(crate) async fn read_http_request(
    stream: &mut TcpStream,
    stop: &CancelToken,
) -> Option<MockHttpRequest> {
    const MAX_MOCK_REQUEST_BYTES: usize = 1024 * 1024;
    let mut buf = vec![0u8; 16 * 1024];
    let mut filled = 0usize;
    let mut header_end = None;
    let mut content_len = 0usize;

    loop {
        let n = tokio::select! {
            biased;
            _ = stop.cancelled() => return None,
            r = stream.read(&mut buf[filled..]) => match r {
                Ok(0) | Err(_) => break,
                Ok(n) => n,
            }
        };
        filled += n;
        if header_end.is_none()
            && let Some(pos) = buf[..filled].windows(4).position(|w| w == b"\r\n\r\n")
        {
            let head = String::from_utf8_lossy(&buf[..pos]);
            for line in head.lines().skip(1) {
                if let Some((k, v)) = line.split_once(':')
                    && k.trim().eq_ignore_ascii_case("content-length")
                {
                    content_len = v.trim().parse().unwrap_or(0);
                }
            }
            header_end = Some(pos + 4);
        }
        if let Some(he) = header_end {
            let needed = he.saturating_add(content_len);
            if filled >= needed {
                break;
            }
            if buf.len() < needed && needed <= MAX_MOCK_REQUEST_BYTES {
                buf.resize(needed, 0);
            }
        }
        if filled == buf.len() {
            break;
        }
    }

    if filled == 0 {
        return None;
    }

    let he = header_end.unwrap_or(filled);
    let head_str = String::from_utf8_lossy(&buf[..he]);
    let first_line = head_str.lines().next().unwrap_or("");
    let path_and_query = first_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_owned();
    let mut headers = Vec::new();
    for line in head_str.lines().skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
        }
    }
    let body_end = filled.min(he.saturating_add(content_len));
    let body = buf[he..body_end].to_vec();

    Some(MockHttpRequest {
        path_and_query,
        headers,
        body,
    })
}

/// Write a `Connection: close` HTTP/1.1 response with an automatic
/// `Content-Length` header.
pub(crate) async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    extra_headers: &str,
    body: &[u8],
) {
    let header = format!(
        "HTTP/1.1 {status} {reason}\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes()).await;
    if !body.is_empty() {
        let _ = stream.write_all(body).await;
    }
    let _ = stream.shutdown().await;
}

/// Loopback TCP server running on a dedicated `current_thread` Tokio runtime
/// with deterministic [`CancelToken`] shutdown on drop.
pub(crate) struct MockHttpServer {
    addr: SocketAddr,
    shutdown: CancelToken,
    handle: Option<thread::JoinHandle<()>>,
}

impl MockHttpServer {
    pub(crate) fn start<F, Fut>(handler: F) -> Self
    where
        F: Fn(MockHttpRequest, TcpStream, CancelToken) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let shutdown = CancelToken::new();
        let stop = shutdown.clone();
        let handler = Arc::new(handler);
        let (addr_tx, addr_rx) = mpsc::channel();

        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("mock http server runtime");
            rt.block_on(async move {
                let listener = TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind loopback");
                let addr = listener.local_addr().expect("local_addr");
                let _ = addr_tx.send(addr);

                loop {
                    let accepted = tokio::select! {
                        biased;
                        _ = stop.cancelled() => break,
                        res = listener.accept() => res,
                    };
                    let Ok((mut stream, _)) = accepted else {
                        break;
                    };
                    let handler_conn = Arc::clone(&handler);
                    let stop_conn = stop.clone();
                    tokio::spawn(async move {
                        let Some(req) = read_http_request(&mut stream, &stop_conn).await else {
                            return;
                        };
                        handler_conn(req, stream, stop_conn).await;
                    });
                }
            });
        });

        let addr = addr_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("mock http server addr");
        Self {
            addr,
            shutdown,
            handle: Some(handle),
        }
    }

    pub(crate) fn addr(&self) -> SocketAddr {
        self.addr
    }
}

impl Drop for MockHttpServer {
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}
