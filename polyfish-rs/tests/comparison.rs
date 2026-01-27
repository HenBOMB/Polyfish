use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::types::{AbilityType, MoveType, RewardType, StructureType, TechnologyType, UnitType};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ComparisonData {
    initialState: serde_json::Value,
    moves: Vec<serde_json::Value>,
    finalState: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct MoveData {
    #[serde(rename = "type")]
    move_type: Option<MoveType>,
    #[serde(default)]
    src: Option<i32>,
    #[serde(default)]
    target: Option<i32>,

    // For specific moves
    structType: Option<StructureType>,
    unitType: Option<UnitType>,
    techType: Option<TechnologyType>,
    abilityType: Option<AbilityType>,
    rewardType: Option<RewardType>,
}

// Helper to convert JSON MoveData to Box<dyn Move>
fn deserialize_move(data: &serde_json::Value) -> Box<dyn Move> {
    // We can't use standard serde deserialization easily because Move is a trait
    // So we manually parse based on move type

    // Check if it matches our struct
    let m: MoveData = serde_json::from_value(data.clone()).unwrap();
    let move_type = m
        .move_type
        .or_else(|| {
            // Fallback for "moveType" property if "type" is missing (TS Move.serialize vs generic JSON)
            data.get("moveType")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
        })
        .unwrap();

    match move_type {
        MoveType::Step => Box::new(polyfish::moves::StepMove::new(
            m.src.unwrap(),
            m.target.unwrap(),
        )),
        MoveType::Attack => Box::new(polyfish::moves::AttackMove::new(
            m.src.unwrap(),
            m.target.unwrap(),
        )),
        MoveType::Ability => {
            let ability_type = m.abilityType.unwrap();
            match ability_type {
                AbilityType::Recover => Box::new(polyfish::moves::RecoverMove::new(m.src.unwrap())),
                AbilityType::Disband => Box::new(polyfish::moves::DisbandMove::new(m.src.unwrap())),
                AbilityType::ClearForest => {
                    Box::new(polyfish::moves::ClearForestMove::new(m.target.unwrap()))
                }
                AbilityType::GrowForest => {
                    Box::new(polyfish::moves::GrowForestMove::new(m.target.unwrap()))
                }
                AbilityType::BurnForest => {
                    Box::new(polyfish::moves::BurnForestMove::new(m.target.unwrap()))
                }
                AbilityType::Destroy => {
                    Box::new(polyfish::moves::DestroyMove::new(m.target.unwrap()))
                }
                AbilityType::Decompose => {
                    Box::new(polyfish::moves::DecomposeMove::new(m.target.unwrap()))
                }
                AbilityType::HealOthers => {
                    Box::new(polyfish::moves::HealOthersMove::new(m.src.unwrap()))
                }
                AbilityType::Boost => Box::new(polyfish::moves::BoostMove::new(m.src.unwrap())),
                AbilityType::Explode => Box::new(polyfish::moves::ExplodeMove::new(m.src.unwrap())),
                AbilityType::Promote => Box::new(polyfish::moves::PromoteMove::new(m.src.unwrap())),
                AbilityType::EnchantAnimal => {
                    Box::new(polyfish::moves::EnchantAnimalMove::new(m.target.unwrap()))
                }
                // Add other abilities as needed if they appear in the test data
                _ => panic!(
                    "Ability type {:?} not implemented in comparison test deserializer",
                    ability_type
                ),
            }
        }
        MoveType::Summon => Box::new(polyfish::moves::SummonMove::new(
            m.src.unwrap(),
            m.unitType.unwrap(),
        )),
        MoveType::Harvest => Box::new(polyfish::moves::HarvestMove::new(m.target.unwrap())),
        MoveType::Build => Box::new(polyfish::moves::BuildMove::new(
            m.target.unwrap(),
            m.structType.unwrap(),
        )),
        MoveType::Research => Box::new(polyfish::moves::ResearchMove::new(m.techType.unwrap())),
        MoveType::Capture => Box::new(polyfish::moves::CaptureMove::new(m.src.unwrap())),
        MoveType::Reward => Box::new(polyfish::moves::RewardMove::new(
            m.src.unwrap(),
            m.rewardType.unwrap(),
        )),
        MoveType::EndTurn => Box::new(polyfish::moves::EndTurnMove),
        _ => panic!(
            "Move type {:?} not implemented in comparison test deserializer",
            move_type
        ),
    }
}

#[test]
fn test_comparison() {
    let path = Path::new("../comparison_data.json");
    if !path.exists() {
        println!("Skipping comparison test: data file not found");
        return;
    }

    let file_content = fs::read_to_string(path).unwrap();
    let data: ComparisonData = serde_json::from_str(&file_content).unwrap();

    // Initialize game with initial state
    let json_str = data.initialState.to_string();
    if json_str.len() > 3106 {
        let start = 3106usize.saturating_sub(50);
        let end = (3106 + 50).min(json_str.len());
        println!("JSON around error: ...{}...", &json_str[start..end]);
    }
    let mut game = Game::from_json(&json_str).expect("Failed to load initial state");

    println!("Loaded game with {} tiles", game.tile_count());
    println!(
        "Loaded game with {} tiles and {} tribes",
        game.tile_count(),
        game.state.tribes.len()
    );
    for (id, tribe) in &game.state.tribes {
        println!(
            "  Tribe {}: killed={}, resigned={}, score={}, stars={}",
            id, tribe.killed_turn, tribe.resigned_turn, tribe.score, tribe.stars
        );
    }

    println!(
        "Game settings: mode={:?}, turn={}, max_turns={}",
        game.state.settings.mode, game.state.settings.turn, game.state.settings.max_turns
    );

    let alive_count = game
        .state
        .tribes
        .values()
        .filter(|t| t.killed_turn <= 0 && t.resigned_turn <= 0)
        .count();
    println!("Alive count (test logic): {}", alive_count);

    for (i, move_json) in data.moves.iter().enumerate() {
        println!("Executing move {}: {:?}", i, move_json);

        let game_move = deserialize_move(move_json);
        let move_type = game_move.move_type();
        let pov_id = game.state.settings.current_player_turn_id;

        // Execute move
        let result = game.play_move(game_move.as_ref());

        if move_type == MoveType::EndTurn {
            let tribe = &game.state.tribes[&pov_id];
            let spt = polyfish::functions::get_tribe_spt(&game.state, tribe);
            println!(
                "  Turn End for Tribe {}: stars={}, SPT={}, current_round={}",
                pov_id, tribe.stars, spt, game.state.settings.turn
            );
        }

        assert!(
            result.is_some(),
            "Move {} failed to execute (Game Over returned None)",
            i
        );

        // TODO: Verify state against expected state if we had it
        // For now, ensuring it runs without panic and returns valid result is the first step.
        // We could also check hash consistency if we implement identical hashing.
    }

    println!(
        "Successfully executed {} moves. Verifying final state...",
        data.moves.len()
    );

    let expected_state: polyfish::states::GameState =
        serde_json::from_value(data.finalState).unwrap();

    for (id, tribe) in &game.state.tribes {
        if let Some(expected_tribe) = expected_state.tribes.get(id) {
            println!(
                "  Verifying Tribe {}: stars={}, score={}, cities={}, units={}",
                id,
                tribe.stars,
                tribe.score,
                tribe.cities.len(),
                tribe.units.len()
            );

            assert_eq!(
                tribe.stars, expected_tribe.stars,
                "Tribe {} stars mismatch",
                id
            );
            // Score might have slight differences, but let's check it anyway
            assert_eq!(
                tribe.score, expected_tribe.score,
                "Tribe {} score mismatch",
                id
            );
            assert_eq!(
                tribe.cities.len(),
                expected_tribe.cities.len(),
                "Tribe {} cities count mismatch",
                id
            );
            assert_eq!(
                tribe.units.len(),
                expected_tribe.units.len(),
                "Tribe {} units count mismatch",
                id
            );
        }
    }

    println!("Final state verification passed!");
}
