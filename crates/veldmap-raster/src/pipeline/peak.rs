//! Named working-memory peak estimation (`DecodePeak`) for analytical raster
//! pyramid level planning and admission checks.

use veldmap_core::budget::BudgetError;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;

/// Format a byte count cleanly without truncating sub-MiB values to `0 MiB`.
fn format_bytes(bytes: u64) -> String {
    if bytes >= MIB {
        format!("{} MiB", bytes.div_ceil(MIB))
    } else if bytes >= KIB {
        format!("{} KiB", bytes.div_ceil(KIB))
    } else {
        format!("{bytes} B")
    }
}

/// Named working-memory peak breakdown for one decode operation or pyramid
/// level row.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct DecodePeak {
    terms: Vec<(&'static str, u64)>,
}

impl DecodePeak {
    /// Construct an empty peak accumulator.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Add a named cost term in bytes.
    pub(super) fn with(mut self, what: &'static str, bytes: u64) -> Self {
        self.terms.push((what, bytes));
        self
    }

    /// Sum of all named terms in bytes (saturating on overflow).
    pub(super) fn total(&self) -> u64 {
        self.terms
            .iter()
            .map(|(_, bytes)| *bytes)
            .fold(0u64, u64::saturating_add)
    }

    /// Return whether `total() <= limit_bytes`.
    #[cfg(test)]
    pub(super) fn fits(&self, limit_bytes: u64) -> bool {
        self.total() <= limit_bytes
    }

    /// Validate that `total() <= limit_bytes`, returning a structured
    /// [`BudgetError::ExceedsCap`] naming every component when refused.
    pub(super) fn admit(&self, limit_bytes: u64) -> Result<(), BudgetError> {
        let requested = self.total();
        if requested <= limit_bytes {
            Ok(())
        } else {
            Err(BudgetError::ExceedsCap {
                requested,
                cap: limit_bytes,
                detail: self.note(),
            })
        }
    }

    /// Human-readable summary of terms (e.g. `"band 128 MiB, tile 256 KiB"`).
    pub(super) fn note(&self) -> String {
        self.terms
            .iter()
            .map(|(what, bytes)| format!("{what} {}", format_bytes(*bytes)))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_peak_admit_and_note_format_mib_kib_and_bytes() {
        let peak = DecodePeak::new()
            .with("band", 4 * MIB)
            .with("tile", 256 * KIB)
            .with("header", 512);
        assert!(peak.admit(5 * MIB).is_ok());
        assert_eq!(peak.note(), "band 4 MiB, tile 256 KiB, header 512 B");
        assert!(matches!(
            peak.admit(4 * MIB),
            Err(BudgetError::ExceedsCap { detail, .. }) if detail.contains("band 4 MiB") && detail.contains("tile 256 KiB")
        ));
    }
}
