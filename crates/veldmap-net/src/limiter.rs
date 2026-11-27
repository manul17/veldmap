//! Per-host HTTP concurrency semaphores and rate-limit cooldown (`429`/`503`
//! `Retry-After`) tracking shared across live basemaps, HTTP Range COGs, and
//! STAC search requests.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, RETRY_AFTER};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use veldmap_core::sync::{CancelToken, lock_unpoisoned};

/// Minimum and maximum clamped `Retry-After` cooldown window.
const MIN_RETRY_AFTER: Duration = Duration::from_secs(1);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(120);

/// Lazily prune idle host entries whenever a new host is inserted while at
/// least this many hosts are tracked, preventing unbounded map growth in long
/// multi-host sessions.
const HOST_LIMITER_PRUNE_THRESHOLD: usize = 64;

struct HostEntry {
    semaphore: Arc<Semaphore>,
    cooldown_until: Option<Instant>,
    cooldown_notify: Arc<Notify>,
}

impl HostEntry {
    fn is_active(&self, max_per_host: usize, now: Instant) -> bool {
        let has_cooldown = self.cooldown_until.is_some_and(|u| u > now);
        has_cooldown
            || Arc::strong_count(&self.semaphore) > 1
            || self.semaphore.available_permits() < max_per_host
            || Arc::strong_count(&self.cooldown_notify) > 1
    }
}

/// Reason a per-host HTTP permit could not be acquired immediately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PermitDenied {
    Cooldown(Instant),
    Saturated,
}

/// Async cancellation marker returned by [`HostLimiters::acquire_permit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PermitCancelled;

/// Per-host concurrency semaphores and rate-limit cooldown tracking (`429` /
/// `503` `Retry-After`).
///
/// Lock ordering: `HostLimiters` is a strict leaf lock — it never acquires
/// external mutexes or signals notifiers while locked, so callers may safely
/// invoke it while holding higher-level scheduler locks.
pub(crate) struct HostLimiters {
    max_per_host: usize,
    map: Mutex<HashMap<String, HostEntry>>,
}

impl HostLimiters {
    pub(crate) fn new(max_per_host: usize) -> Self {
        Self {
            max_per_host: max_per_host.max(1),
            map: Mutex::new(HashMap::new()),
        }
    }

    fn with_entry<R>(&self, host: &str, f: impl FnOnce(&mut HostEntry) -> R) -> R {
        let mut map = lock_unpoisoned(&self.map);
        if let Some(existing) = map.get_mut(host) {
            return f(existing);
        }
        if map.len() >= HOST_LIMITER_PRUNE_THRESHOLD {
            let now = Instant::now();
            let max_per_host = self.max_per_host;
            map.retain(|_, entry| entry.is_active(max_per_host, now));
        }
        let entry = map.entry(host.to_owned()).or_insert_with(|| HostEntry {
            semaphore: Arc::new(Semaphore::new(self.max_per_host)),
            cooldown_until: None,
            cooldown_notify: Arc::new(Notify::new()),
        });
        f(entry)
    }

    /// Atomically check host cooldown and attempt to acquire a per-host HTTP
    /// permit under a single lock, avoiding string allocation on existing host
    /// hits.
    pub(crate) fn try_acquire_permit(
        &self,
        host: &str,
        now: Instant,
    ) -> Result<OwnedSemaphorePermit, PermitDenied> {
        self.with_entry(host, |entry| {
            if let Some(until) = entry.cooldown_until.filter(|&u| u > now) {
                return Err(PermitDenied::Cooldown(until));
            }
            Arc::clone(&entry.semaphore)
                .try_acquire_owned()
                .map_err(|_| PermitDenied::Saturated)
        })
    }

    /// Asynchronously wait out any active host cooldown and acquire a per-host
    /// HTTP concurrency permit, aborting immediately if `cancel` triggers.
    pub(crate) async fn acquire_permit(
        &self,
        host: &str,
        cancel: &CancelToken,
    ) -> Result<OwnedSemaphorePermit, PermitCancelled> {
        loop {
            if cancel.is_cancelled() {
                return Err(PermitCancelled);
            }
            let cooldown_notify = self.with_entry(host, |entry| Arc::clone(&entry.cooldown_notify));
            let notified = cooldown_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            let (semaphore, cooldown_until) = self.with_entry(host, |entry| {
                (Arc::clone(&entry.semaphore), entry.cooldown_until)
            });

            if let Some(until) = cooldown_until.filter(|&u| u > Instant::now()) {
                let deadline = tokio::time::Instant::from_std(until);
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(PermitCancelled),
                    _ = notified => continue,
                    _ = tokio::time::sleep_until(deadline) => continue,
                }
            }

            let permit = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(PermitCancelled),
                res = semaphore.acquire_owned() => match res {
                    Ok(permit) => permit,
                    Err(_) => return Err(PermitCancelled),
                },
            };

            // Re-check whether a `429`/`503` cooldown started while we waited
            // on the semaphore; if so, release the permit and honor cooldown.
            let active_cooldown = self.with_entry(host, |entry| {
                entry.cooldown_until.filter(|&u| u > Instant::now())
            });
            if active_cooldown.is_some() {
                drop(permit);
                continue;
            }
            return Ok(permit);
        }
    }

    /// Record or extend a per-host cooldown window (e.g. on HTTP `429`/`503`),
    /// avoiding string allocation when `host` is already tracked.
    pub(crate) fn record_cooldown(&self, host: &str, until: Instant) {
        self.with_entry(host, |entry| {
            entry.cooldown_until = Some(match entry.cooldown_until {
                Some(existing) if existing > until => existing,
                _ => until,
            });
        });
    }

    /// Clear cooldowns strictly for `hosts` (leaving unrelated hosts untouched)
    /// and prune idle entries with no active permits or active cooldowns.
    pub(crate) fn clear_cooldowns_for<'a>(&self, hosts: impl IntoIterator<Item = &'a str>) {
        let mut to_notify = Vec::new();
        {
            let mut map = lock_unpoisoned(&self.map);
            let now = Instant::now();
            for host in hosts {
                if let Some(entry) = map.get_mut(host)
                    && entry.cooldown_until.take().is_some()
                {
                    to_notify.push(Arc::clone(&entry.cooldown_notify));
                }
            }
            let max_per_host = self.max_per_host;
            map.retain(|_, entry| entry.is_active(max_per_host, now));
        }
        // Notify outside `self.map` so `HostLimiters` remains a strict leaf lock.
        for notify in to_notify {
            notify.notify_waiters();
        }
    }
}

pub(crate) fn parse_retry_after_seconds(header: Option<&str>) -> Option<Duration> {
    let raw = header?.trim();
    let secs = raw.parse::<u64>().ok()?;
    Some(Duration::from_secs(secs).clamp(MIN_RETRY_AFTER, MAX_RETRY_AFTER))
}

/// Record a host cooldown from an HTTP `429`/`503` response's `Retry-After`
/// header (or `default_backoff` if missing/unparseable).
pub(crate) fn record_rate_limit_cooldown(
    limiters: &HostLimiters,
    host: &str,
    headers: &HeaderMap,
    default_backoff: Duration,
) {
    let retry_after = headers
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| parse_retry_after_seconds(Some(s)))
        .unwrap_or(default_backoff);
    limiters.record_cooldown(host, Instant::now() + retry_after);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_and_host_limiters_enforce_cooldowns_and_permits() {
        assert_eq!(
            parse_retry_after_seconds(Some("0")),
            Some(Duration::from_secs(1))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("30")),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("86400")),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            parse_retry_after_seconds(Some("Wed, 21 Oct 2026 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after_seconds(None), None);

        let limiters = HostLimiters::new(1);
        let now = Instant::now();
        let permit = limiters
            .try_acquire_permit("example.org", now)
            .expect("first permit");
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Saturated)
        );
        drop(permit);

        let later = now + Duration::from_secs(30);
        limiters.record_cooldown("example.org", later);
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Cooldown(later))
        );
        // Shorter cooldown must not shorten an existing longer cooldown.
        limiters.record_cooldown("example.org", now + Duration::from_secs(5));
        assert_eq!(
            limiters.try_acquire_permit("example.org", now).err(),
            Some(PermitDenied::Cooldown(later))
        );

        limiters.record_cooldown("other.org", later);
        limiters.clear_cooldowns_for(["example.org"]);
        assert!(limiters.try_acquire_permit("example.org", now).is_ok());
        assert_eq!(
            limiters.try_acquire_permit("other.org", now).err(),
            Some(PermitDenied::Cooldown(later)),
            "clearing example.org must leave unrelated host cooldowns intact"
        );

        // Inserting past the prune threshold lazily evicts idle hosts while
        // keeping hosts with an active permit or active cooldown.
        let held_permit = limiters
            .try_acquire_permit("busy.org", now)
            .expect("busy permit");
        for i in 0..HOST_LIMITER_PRUNE_THRESHOLD {
            let host = format!("idle-{i}.example.org");
            let p = limiters
                .try_acquire_permit(&host, now)
                .expect("idle permit");
            drop(p);
        }
        let trigger = limiters
            .try_acquire_permit("trigger.example.org", now)
            .expect("trigger permit");
        drop(trigger);
        {
            let map = lock_unpoisoned(&limiters.map);
            assert!(
                map.len() < HOST_LIMITER_PRUNE_THRESHOLD,
                "idle entries should be pruned lazily, got {}",
                map.len()
            );
            assert!(map.contains_key("busy.org"));
            assert!(map.contains_key("other.org"));
            assert!(map.contains_key("trigger.example.org"));
        }
        drop(held_permit);
    }
}
