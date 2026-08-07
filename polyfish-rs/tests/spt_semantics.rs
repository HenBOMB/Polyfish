//! SPT fidelity (Aug 2026): Workshop/Park count once (+1 SPT each, via the
//! derived rewards count only), Market adjacency counts friendly hubs only,
//! and Greedy's economy evaluator reads the derived production.

use polyfish::functions::get_city_production;
use polyfish::moves::Move;
use polyfish::moves::reward::RewardMove;
use polyfish::states::{CityState, GameState, StructureState, TileState, TribeState};
use polyfish::types::{CityRewardType, StructureType};

fn state_with_city(level: i32) -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    state.settings.current_player_turn_id = 1;
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.stars = 20;
    let mut city = CityState::default();
    city.idx = 60;
    city.owner = 1;
    city.level = level;
    city.production = level; // base + level-ups, no reward bumps
    tribe.cities.push(city);
    state.tribes.insert(1, tribe);
    state.tiles.entry(60).or_insert_with(TileState::default).owner = 1;
    state
}

#[test]
fn workshop_counts_once_in_city_production() {
    let mut state = state_with_city(2);
    let before = {
        let c = &state.tribes.get(&1).unwrap().cities[0];
        get_city_production(&state, c)
    };
    let mv = RewardMove::new(60, CityRewardType::Workshop);
    mv.execute(&mut state).unwrap();
    let after = {
        let c = &state.tribes.get(&1).unwrap().cities[0];
        get_city_production(&state, c)
    };
    assert_eq!(after - before, 1, "Workshop must add exactly +1 SPT");
}

#[test]
fn park_gives_score_and_single_spt() {
    let mut state = state_with_city(5);
    let (spt_before, score_before) = {
        let t = state.tribes.get(&1).unwrap();
        (get_city_production(&state, &t.cities[0]), t.score)
    };
    let mv = RewardMove::new(60, CityRewardType::Park);
    mv.execute(&mut state).unwrap();
    let (spt_after, score_after) = {
        let t = state.tribes.get(&1).unwrap();
        (get_city_production(&state, &t.cities[0]), t.score)
    };
    assert_eq!(spt_after - spt_before, 1, "Park must add exactly +1 SPT");
    assert_eq!(score_after - score_before, 250, "Park must add +250 score");
}

#[test]
fn market_ignores_enemy_owned_adjacent_hubs() {
    let mut state = state_with_city(2);
    // Market at 59 in the city's territory; windmills at 58 (friendly) and
    // 48 (enemy-owned tile). Each windmill gets one adjacent Farm on a tile its
    // own side owns, so both are level 1 — the only difference is ownership.
    state.tribes.get_mut(&1).unwrap().cities[0]._territory.push(59);
    for (idx, owner) in [(59, 1), (58, 1), (57, 1), (48, 2), (37, 2)] {
        state.tiles.entry(idx).or_insert_with(TileState::default).owner = owner;
    }
    for (idx, st) in [
        (59, StructureType::Market),
        (58, StructureType::Windmill),
        (57, StructureType::Farm),
        (48, StructureType::Windmill),
        (37, StructureType::Farm),
    ] {
        state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: st,
                ..Default::default()
            }),
        );
    }
    // base 2 (level) + the friendly windmill's level of 1; the enemy windmill is
    // also level 1 but sits on an enemy tile, so it pays nothing.
    let c = &state.tribes.get(&1).unwrap().cities[0];
    assert_eq!(get_city_production(&state, c), 3);
}

#[test]
fn evaluate_economy_counts_market_and_capital_income() {
    let mut state = state_with_city(2);
    let base = polyfish::ai::evaluator::economy::evaluate_economy(&state, 1);
    // Making the city tile the capital adds derived income the raw
    // `city.production` field never sees.
    state.tiles.get_mut(&60).unwrap().capital_of = 1;
    let with_capital = polyfish::ai::evaluator::economy::evaluate_economy(&state, 1);
    assert!(
        with_capital > base,
        "capital bonus must raise the economy score ({with_capital} vs {base})"
    );
}
