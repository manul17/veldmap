//! Small generic single-thread background worker (`SingleThreadWorker`) shared
//! by off-UI-thread local file I/O workers (`OverlayIoWorker`,
//! `VectorIoWorker`) while preserving distinct named OS threads and
//! deterministic Phase 1 `close_channel` / Phase 3 `join` shutdown ordering.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Single-thread command/result background worker with explicit two-phase
/// shutdown (`close_channel` then `join`).
pub(crate) struct SingleThreadWorker<Cmd, Res> {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Res>,
    handle: Option<JoinHandle<()>>,
}

impl<Cmd, Res> SingleThreadWorker<Cmd, Res>
where
    Cmd: Send + 'static,
    Res: Send + 'static,
{
    /// Spawn a named background thread running `worker_loop(cmd_rx, res_tx)`.
    ///
    /// If OS thread creation fails, logs an error and leaves the command
    /// channel closed so callers receive `false` from [`Self::send`].
    pub(crate) fn spawn(
        thread_name: &str,
        worker_loop: impl FnOnce(Receiver<Cmd>, Sender<Res>) + Send + 'static,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        let (res_tx, res_rx) = mpsc::channel::<Res>();
        let handle = match thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || worker_loop(cmd_rx, res_tx))
        {
            Ok(h) => Some(h),
            Err(err) => {
                tracing::error!(
                    thread = thread_name,
                    error = %err,
                    "Failed to spawn background worker"
                );
                None
            }
        };

        Self {
            tx: handle.is_some().then_some(cmd_tx),
            rx: res_rx,
            handle,
        }
    }
}

impl<Cmd, Res> SingleThreadWorker<Cmd, Res> {
    pub(crate) fn send(&self, cmd: Cmd) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(cmd).is_ok())
    }

    pub(crate) fn try_recv(&self) -> Option<Res> {
        self.rx.try_recv().ok()
    }

    /// Phase 1 shutdown: close the command channel so the worker loop exits
    /// after completing or cancelling any active item.
    pub(crate) fn close_channel(&mut self) {
        self.tx = None;
    }

    /// Phase 3 shutdown: close the command channel (if still open) and join
    /// the background thread deterministically.
    pub(crate) fn join(&mut self) {
        self.tx = None;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl<Cmd, Res> Drop for SingleThreadWorker<Cmd, Res> {
    fn drop(&mut self) {
        self.join();
    }
}
