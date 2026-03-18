use polyfish::ai::evaluator;
use polyfish::states::{CityState, GameState, TribeState, UnitState};
use polyfish::types::{TechnologyType, TribeType, UnitType};

fn setup_state() -> (GameState, i32) {
    let mut state = GameState::default();
    state.settings.size = 11;
    state.settings.max_turns = 30;
    state.settings.turn = 5; // Early game

    let tribe_id = 1;
    let mut tribe = TribeState::default();
    tribe.id = tribe_id;
    tribe.tribe_type = TribeType::Imperius;
    tribe.stars = 5;

    // Add a capital city
    let mut city = CityState::default();
    city.production = 2; // +1 base = 3 income
    city.level = 2;
    tribe.cities.push(city);

    state.tribes.insert(tribe_id, tribe);
    (state, tribe_id)
}

#[test]
fn preview_evaluator() {
    let (mut state, params) = setup_state();
    let player_id = params;

    let genes = polyfish::ai::genes::AIGenes::default();
    let score_base = evaluator::evaluate_state(&state, player_id, &genes);
    println!("Base Score (Turn 5): {:.4}", score_base);

    // 1. Add Economy
    if let Some(t) = state.tribes.get_mut(&player_id) {
        t.stars += 10;
        t.tech_vanilla.push(polyfish::states::TechnologyState {
            tech_type: TechnologyType::Riding,
            discovered: true,
        });
    }
    let score_eco = evaluator::evaluate_state(&state, player_id, &genes);
    println!("Score with +10 Stars + Tech: {:.4}", score_eco);
    assert!(score_eco > score_base);

    // 2. Add Military
    if let Some(t) = state.tribes.get_mut(&player_id) {
        let mut unit = UnitState::default();
        unit.unit_type = UnitType::Warrior;
        unit.health = 100; // max
        t.units.push(unit);
    }
    let score_mil = evaluator::evaluate_state(&state, player_id, &genes);
    println!("Score with +1 Warrior: {:.4}", score_mil);
    assert!(score_mil > score_eco);

    // 3. Late Game Weighting
    state.settings.turn = 25; // Late game
    let score_late = evaluator::evaluate_state(&state, player_id, &genes);
    println!("Score Late Game (Turn 25): {:.4}", score_late);

    // In late game, military should be weighted higher.
    // Our army is weak (1 warrior), output might be lower or higher depending on weights.
    // Early weights: Eco 0.5, Exp 0.3, Mil 0.2
    // Late weights: Eco 0.2, Exp 0.1, Mil 0.7
    // Eco score is decent, Mil score is low. So Late score should be LOWER than Early score for this specific state.
    // Eco Score ≈ (3/40)*0.6 + (15/40)*0.2 + (1/25)*0.2 ≈ 0.045 + 0.075 + 0.008 ≈ 0.128
    // Mil Score ≈ (0.55 / 20) ≈ 0.027
    // Early: 0.128*0.5 + ... + 0.027*0.2 ≈ 0.064 + 0.005 ≈ 0.069
    // Late: 0.128*0.2 + ... + 0.027*0.7 ≈ 0.025 + 0.019 ≈ 0.044
    // So Late < Early.

    // 4. Relative Scoring Check
    // Create a dummy opponent
    let opponent_id = 2;
    let mut opponent = TribeState::default();
    opponent.id = opponent_id;
    opponent.stars = 0; // Very poor opponent
    state.tribes.insert(opponent_id, opponent);

    let relative_score = evaluator::evaluate_state(&state, player_id, &genes);
    println!("Relative Score vs Poor Opponent: {:.4}", relative_score);
    // My absolute score ~0.05. Opponent ~0.0. Relative ~0.05.
    assert!(relative_score > 0.0);

    // Buff opponent
    if let Some(op) = state.tribes.get_mut(&opponent_id) {
        op.stars = 100; // Rich opponent
        // Add huge army
        for _ in 0..20 {
            let mut unit = UnitState::default();
            unit.unit_type = UnitType::Giant; // Strong unit
            op.units.push(unit);
        }
    }

    let relative_score_losing = evaluator::evaluate_state(&state, player_id, &genes);
    println!(
        "Relative Score vs Rich Opponent: {:.4}",
        relative_score_losing
    );
    // Opponent has max score (1.0). My score ~0.05. Relative ~ -0.95.
    assert!(relative_score_losing < 0.0);

    // 5. Exploration Check
    // Reveal a tile
    if let Some(tile) = state.tiles.get_mut(&0) {
        tile.explorers.insert(player_id);
    }

    // Let's explore 30 tiles.
    for i in 0..30 {
        if let Some(tile) = state.tiles.get_mut(&i) {
            tile.explorers.insert(player_id);
        }
    }

    // Let's go back to Early Game
    state.settings.turn = 5;
    let index_score_early = evaluator::player::evaluate_player(&state, player_id, &genes);
    println!(
        "Score Early Game (Turn 5) with 30 tiles explored: {:.4}",
        index_score_early
    );

    // Base Early (Turn 5) was 0.08.
    // Exploration:
    // Total = 121. Explored = 31 (0+30).
    // Min = 24.2 (20%). Spread = 72.6 (60%).
    // Score = (31 - 24.2) / 72.6 = 6.8 / 72.6 ≈ 0.0936.
    // Weight = 0.2.
    // Increment = 0.0936 * 0.2 = 0.0187.
    // Expected > 0.08 + 0.018 = 0.098.
    assert!(index_score_early > 0.09);
}
