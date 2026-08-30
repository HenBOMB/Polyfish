//! Split to a separate file so eval_seeds.rs stays one readable unit.
//!
//! Was `seed_selection_tests` inside src/bin/self_play.rs. arena carried a
//! near-identical copy that never ran -- arena is `test = false` in
//! Cargo.toml, so nothing ever compiled it. This is now the only copy, and
//! it runs under `cargo test --lib`.

use super::*;
use crate::types::TribeType;

    #[test]
    fn no_seed_file_derives_base_seed_plus_i_unchanged() {
        for i in 0..5usize {
            assert_eq!(seed_for_game(i, 1787300000, None), (1787300000u64 + i as u64) as i64);
        }
    }

    #[test]
    fn seed_file_uses_exact_listed_seeds_not_the_derived_sequence() {
        let list = vec![42i64, 9001, 7, 123456789];
        for (i, &expected) in list.iter().enumerate() {
            let got = seed_for_game(i, 1787300000, Some(&list));
            assert_eq!(got, expected);
            // Distinct from what base_seed + i would have produced, so this
            // is actually exercising the fixed list, not coincidentally
            // matching the legacy derivation.
            assert_ne!(got, (1787300000u64 + i as u64) as i64);
        }
    }

    #[test]
    fn seed_file_shorter_than_game_count_errors_loudly() {
        let tmp = std::env::temp_dir().join(format!("polyfish_seed_file_test_{}.json", std::process::id()));
        std::fs::write(&tmp, r#"{"seeds": [{"seed": 1}, {"seed": 2}, {"seed": 3}]}"#).unwrap();
        let result = load_seed_file(tmp.to_str().unwrap(), 4, parse_tribe);
        std::fs::remove_file(&tmp).ok();
        assert!(result.is_err(), "requesting more games than seeds must error, not wrap");
    }

    #[test]
    fn seed_file_loads_seeds_in_file_order() {
        let tmp = std::env::temp_dir().join(format!("polyfish_seed_file_test_ok_{}.json", std::process::id()));
        std::fs::write(&tmp, r#"{"seeds": [{"seed": 10}, {"seed": 20}, {"seed": 30}]}"#).unwrap();
        let result = load_seed_file(tmp.to_str().unwrap(), 3, parse_tribe).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(result.iter().map(|e| e.seed).collect::<Vec<i64>>(), vec![10, 20, 30]);
        assert!(result.iter().all(|e| e.tribes.is_none()), "entries without tribe1/tribe2 must parse to None");
    }

    #[test]
    fn seed_file_parses_per_entry_tribe_pair() {
        let tmp = std::env::temp_dir().join(format!("polyfish_seed_file_test_tribes_{}.json", std::process::id()));
        std::fs::write(
            &tmp,
            r#"{"seeds": [{"seed": 10, "tribe1": "XinXi", "tribe2": "Zebasi"}, {"seed": 20}]}"#,
        )
        .unwrap();
        let result = load_seed_file(tmp.to_str().unwrap(), 2, parse_tribe).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(result[0].tribes, Some((TribeType::XinXi, TribeType::Zebasi)));
        assert_eq!(result[1].tribes, None);
    }

    #[test]
    fn seed_file_one_sided_tribe_pair_errors_loudly() {
        let tmp = std::env::temp_dir().join(format!("polyfish_seed_file_test_onesided_{}.json", std::process::id()));
        std::fs::write(&tmp, r#"{"seeds": [{"seed": 10, "tribe1": "XinXi"}]}"#).unwrap();
        let result = load_seed_file(tmp.to_str().unwrap(), 1, parse_tribe);
        std::fs::remove_file(&tmp).ok();
        assert!(result.is_err(), "one of tribe1/tribe2 set without the other must error, not silently drop it");
    }

    // resolve_tribes' three-tier precedence: CLI --tribe1/--tribe2 > a
    // --seed-file entry's own tribe pair > pick_tribes' random draw.
    use rand::SeedableRng;

    #[test]
    fn resolve_tribes_cli_pin_beats_seed_file() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let all = vec![TribeType::Imperius, TribeType::Bardur, TribeType::Oumaji];
        let seed_file_pair = Some((TribeType::XinXi, TribeType::Zebasi));
        let got = resolve_tribes(
            &mut rng,
            &all,
            &Some("Bardur".to_string()),
            &Some("Oumaji".to_string()),
            seed_file_pair,
        );
        // Fully-pinned CLI wins outright -- the seed-file pair is ignored,
        // not merged in.
        assert_eq!(got, (TribeType::Bardur, TribeType::Oumaji));
    }

    #[test]
    fn resolve_tribes_seed_file_wins_when_no_cli_pin() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let all = vec![TribeType::Imperius, TribeType::Bardur, TribeType::Oumaji];
        let seed_file_pair = Some((TribeType::XinXi, TribeType::Zebasi));
        let got = resolve_tribes(&mut rng, &all, &None, &None, seed_file_pair);
        assert_eq!(got, (TribeType::XinXi, TribeType::Zebasi));
    }

    #[test]
    fn resolve_tribes_falls_back_to_random_pick_tribes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let all = vec![TribeType::Imperius, TribeType::Bardur, TribeType::Oumaji];
        let got = resolve_tribes(&mut rng, &all, &None, &None, None);
        assert_ne!(got.0, got.1, "pick_tribes never draws a mirror match");
        assert!(all.contains(&got.0) && all.contains(&got.1));
    }
