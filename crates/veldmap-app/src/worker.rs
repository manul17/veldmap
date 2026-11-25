//! Small generic single-thread background worker primitives
//! ([`CommandWorker`] and [`SingleThreadWorker`]) shared by off-UI-thread
//! workers (`DiskCacheWorker`, overlay I/O, and vector I/O) while preserving
//! distinct named OS threads and deterministic Phase 1 `close_channel` /
//! Phase 3 `join` shutdown ordering.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Typed failure returned by [`CommandWorker::send`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CommandSendError {
    /// Command channel was already closed (e.g. during shutdown or because the
    /// worker thread failed to spawn).
    Closed,
    /// Worker thread exited and dropped its receiving end before this command
    /// could be delivered.
    Disconnected,
}

/// Single-thread command-only background worker with explicit two-phase
/// shutdown (`close_channel` then `join`).
pub(crate) struct CommandWorker<Cmd> {
    tx: Option<Sender<Cmd>>,
    handle: Option<JoinHandle<()>>,
}

impl<Cmd> CommandWorker<Cmd>
where
    Cmd: Send + 'static,
{
    /// Attempt to spawn a named background thread running
    /// `worker_loop(cmd_rx)`. Returns `None` if OS thread creation fails.
    pub(crate) fn try_spawn(
        thread_name: &str,
        worker_loop: impl FnOnce(Receiver<Cmd>) + Send + 'static,
    ) -> Option<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
        match thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || worker_loop(cmd_rx))
        {
            Ok(handle) => Some(Self {
                tx: Some(cmd_tx),
                handle: Some(handle),
            }),
            Err(err) => {
                tracing::error!(
                    thread = thread_name,
                    error = %err,
                    "Failed to spawn background worker"
                );
                None
            }
        }
    }

    fn closed() -> Self {
        Self {
            tx: None,
            handle: None,
        }
    }
}

impl<Cmd> CommandWorker<Cmd> {
    /// Enqueue `cmd` onto the background worker, distinguishing an explicitly
    /// closed channel ([`CommandSendError::Closed`]) from a vanished receiver
    /// ([`CommandSendError::Disconnected`]).
    pub(crate) fn send(&self, cmd: Cmd) -> Result<(), CommandSendError> {
        let Some(tx) = &self.tx else {
            return Err(CommandSendError::Closed);
        };
        tx.send(cmd).map_err(|_| CommandSendError::Disconnected)
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

impl<Cmd> Drop for CommandWorker<Cmd> {
    fn drop(&mut self) {
        self.join();
    }
}

/// Single-thread command/result background worker built over [`CommandWorker`]
/// with explicit two-phase shutdown (`close_channel` then `join`).
pub(crate) struct SingleThreadWorker<Cmd, Res> {
    worker: CommandWorker<Cmd>,
    rx: Receiver<Res>,
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
        let (res_tx, res_rx) = mpsc::channel::<Res>();
        let worker = CommandWorker::try_spawn(thread_name, move |cmd_rx| {
            worker_loop(cmd_rx, res_tx);
        })
        .unwrap_or_else(CommandWorker::closed);
        Self { worker, rx: res_rx }
    }
}

impl<Cmd, Res> SingleThreadWorker<Cmd, Res> {
    pub(crate) fn send(&self, cmd: Cmd) -> bool {
        self.worker.send(cmd).is_ok()
    }

    pub(crate) fn try_recv(&self) -> Option<Res> {
        self.rx.try_recv().ok()
    }

    /// Phase 1 shutdown: close the command channel so the worker loop exits
    /// after completing or cancelling any active item.
    pub(crate) fn close_channel(&mut self) {
        self.worker.close_channel();
    }

    /// Phase 3 shutdown: close the command channel (if still open) and join
    /// the background thread deterministically.
    pub(crate) fn join(&mut self) {
        self.worker.join();
    }
}
