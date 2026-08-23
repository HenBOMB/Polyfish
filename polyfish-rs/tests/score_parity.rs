//! Score-parity probe (#40): the incrementally maintained `tribe.score` is the
//! reward/value currency for both TD labels and the reward-aware backup, but
//! the canonical recompute only runs at `post_load`. This walks random
//! playouts and attributes every divergence to the move that introduced it.

use polyfish::game::Game;
use polyfish::score::score_drift;
use polyfish::types::{MapSize, MapType, MoveType, TribeType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const MAP_TYPES: [MapType; 3] = [MapType::Drylands, MapType::Continents, MapType::Archipelago];
const TRIBE_PAIRS: [[TribeType; 2]; 3] = [
    [TribeType::Imperius, TribeType::Bardur],
    [TribeType::Aquarion, TribeType::Polaris],
    [TribeType::Luxidoor, TribeType::Elyrion],
];

/// Play random moves, reporting the first move that changes each tribe's
/// drift. Returns (move description, drift delta) per divergence introduced.
fn probe_playout(
    game_seed: i64,
    steps: usize,
    max_turns: i32,
    conquest: bool,
) -> Vec<(String, i32)> {
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MAP_TYPES[game_seed.unsigned_abs() as usize % MAP_TYPES.len()],
        tribes: TRIBE_PAIRS[game_seed.unsigned_abs() as usize % TRIBE_PAIRS.len()].to_vec(),
        seed: game_seed,
        ..Default::default()
    };
    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    let mut rng = StdRng::seed_from_u64(game_seed as u64);
    let mut prev: HashMap<i32, i32> = score_drift(&game.state).into_iter().collect();
    let mut found = Vec::new();

    for _ in 0..steps {
        if game.state.settings._game_over {
            break;
        }
        let moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        // Random play almost never conquers, and city capture is the drift
        // source with the widest blast radius — so bias toward it explicitly.
        let pick = match conquest {
            true => moves
                .iter()
                .position(|m| {
                    matches!(
                        m.move_type(),
                        MoveType::Capture | MoveType::Attack | MoveType::Summon
                    )
                })
                .unwrap_or_else(|| rng.random_range(0..moves.len())),
            false => rng.random_range(0..moves.len()),
        };
        let m = &moves[pick];
        let desc = m.describe(&game.state);
        if game.play_move(m.as_ref()).is_none() {
            break;
        }
        let now: HashMap<i32, i32> = score_drift(&game.state).into_iter().collect();
        for (&id, &delta) in &now {
            let before = prev.get(&id).copied().unwrap_or(0);
            if delta != before {
                found.push((desc.clone(), delta - before));
            }
        }
        for (&id, &before) in &prev {
            if !now.contains_key(&id) && before != 0 {
                found.push((desc.clone(), -before));
            }
        }
        prev = now;
    }
    found
}

/// The parity gate: over short random playouts, `tribe.score` must equal the
/// canonical recompute at every step.
#[test]
fn test_score_parity_random_playouts() {
    let mut failures = Vec::new();
    for game_seed in 0..6i64 {
        for conquest in [false, true] {
            for (desc, delta) in probe_playout(game_seed, 120, 12, conquest) {
                failures.push(format!(
                    "seed {game_seed} (conquest={conquest}): [{desc}] drifted score by {delta}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "incremental score diverged from calculate_detailed_tribe_score:\n  {}",
        failures.join("\n  ")
    );
}

// Wide probe — run on demand:
//   cargo test --release --test score_parity -- --ignored
#[test]
#[ignore]
fn probe_score_parity_wide() {
    let mut by_move: HashMap<String, (usize, i32, Vec<String>)> = HashMap::new();
    for game_seed in 0..120i64 {
        for (desc, delta) in probe_playout(game_seed, 400, 30, game_seed % 2 == 0) {
            let key = desc.split_whitespace().next().unwrap_or(&desc).to_string();
            let e = by_move.entry(key).or_insert((0, 0, Vec::new()));
            e.0 += 1;
            e.1 += delta.abs();
            if e.2.len() < 3 {
                e.2.push(format!("seed{game_seed} {desc} ({delta:+})"));
            }
        }
    }
    let mut rows: Vec<_> = by_move.into_iter().collect();
    rows.sort_by_key(|(_, (n, _, _))| std::cmp::Reverse(*n));
    eprintln!("score drift by move kind (occurrences, total |delta|):");
    for (kind, (n, total, examples)) in &rows {
        eprintln!(
            "  {kind:<24} {n:>5}  {total:>8}   e.g. {}",
            examples.join(" | ")
        );
    }
    assert!(rows.is_empty(), "score drift observed (see report above)");
}

/// The two halves of the score have to price the same things the same way, so
/// pin what the shared helpers pay out.
#[test]
fn test_score_helpers_price_structures_and_parks() {
    use polyfish::states::{CityState, StructureState};
    use polyfish::types::{CityRewardType, StructureType};

    let temple = |level| StructureState {
        structure_type: StructureType::Temple,
        level,
        founded: 0,
    };
    assert_eq!(polyfish::score::get_structure_score(&temple(1)), 100);
    assert_eq!(polyfish::score::get_structure_score(&temple(3)), 300);
    assert_eq!(
        polyfish::score::get_structure_score(&StructureState {
            structure_type: StructureType::AltarOfPeace,
            level: 1,
            founded: 0,
        }),
        400
    );

    let mut city = CityState {
        idx: 0,
        level: 1,
        population: 0,
        ..Default::default()
    };
    let base = polyfish::score::get_city_transfer_score(&city);
    // Park is re-offered at every level above 4, so it stacks.
    city.rewards.push(CityRewardType::Park);
    city.rewards.push(CityRewardType::Park);
    assert_eq!(polyfish::score::get_city_transfer_score(&city), base + 500);
}
