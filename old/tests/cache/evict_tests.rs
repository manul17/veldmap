// Extracted from master:veldmodules/tile-cache/src/evict.rs

#[cfg(test)]
mod tests {
    use super::*;

    fn source(key: &str, bytes: u64, used: i64) -> Source {
        Source { key: key.to_string(), bytes, used, files: Vec::new() }
    }

    /// Никого не трогали — обычный случай теста про свежесть.
    fn none() -> std::collections::HashSet<String> {
        std::collections::HashSet::new()
    }

    #[test]
    fn under_budget_keeps_everything() {
        let picked = victims(vec![source("a", 10, 1), source("b", 20, 2)], 30, 100, &none());
        assert!(picked.is_empty());
    }

    #[test]
    fn evicts_oldest_first_down_to_headroom() {
        // Бюджет 100, занято 150: цель 90, старейшие «a» и «b» дают 150 → 90,
        // и на этом остановка — «c» свежее и выживает.
        let picked = victims(
            vec![source("c", 50, 30), source("a", 40, 10), source("b", 20, 20), source("d", 40, 40)],
            150,
            100,
            &none(),
        );
        let keys: Vec<&str> = picked.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, vec!["a", "b"]);
    }

    /// Источник, которого трогали в этом окне, не жертва ни при какой
    /// свежести маркера: маркер мог ещё не записаться, а тайлы у него просят
    /// прямо сейчас.
    #[test]
    fn a_touched_source_is_never_a_victim() {
        let alive: std::collections::HashSet<String> = ["hot".to_string()].into_iter().collect();
        let picked = victims(
            // «hot» самый старый по маркеру и самый большой — и всё равно жив.
            vec![source("hot", 90, 1), source("old", 30, 2)],
            120,
            100,
            &alive,
        );
        let keys: Vec<&str> = picked.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, vec!["old"]);
    }

    #[test]
    fn fresh_marker_saves_active_source() {
        // Активный источник (used свежий) выживает, даже если он самый большой.
        let picked = victims(
            vec![source("hot", 90, 100), source("old", 30, 1)],
            120,
            100,
            &none(),
        );
        let keys: Vec<&str> = picked.iter().map(|s| s.key.as_str()).collect();
        assert_eq!(keys, vec!["old"]);
    }
}
