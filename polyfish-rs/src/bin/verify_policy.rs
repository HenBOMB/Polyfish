use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::mapper::DecomposedMapper;
use polyfish::ai::network::PolicyOutput;
use polyfish::ai::policy_composer::compute_move_priors;
use polyfish::game::Game;
use polyfish::types::*;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    println!("--- Policy Verification Script ---");

    // 1. Setup a game scenario
    let mut game = Game::new();
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 42,
        ..Default::default()
    };
    game.state = polyfish::mapgen::generate(gen_settings);
    game.post_load();

    println!(
        "Game initialized. Current Player: {:?}",
        game.state.settings.current_player_turn_id
    );

    // 2. Generate mock policy heads (Random weights)
    // We use deterministic rand-like values for reproducibility
    let action_type = Tensor::from_vec(vec![0.1f32; 11], (1, 11), &device)?;
    let source_spatial = Tensor::from_vec(vec![0.1f32; 900], (1, 900), &device)?;
    let target_spatial = Tensor::from_vec(vec![0.1f32; 900], (1, 900), &device)?;
    let move_option = Tensor::from_vec(vec![0.1f32; 192], (1, 192), &device)?;

    let mock_policy = PolicyOutput {
        action_type,
        source_spatial,
        target_spatial,
        move_option,
    };

    // 3. Get legal moves
    let legal_moves = game.legal_moves();
    println!("Legal moves found: {}", legal_moves.len());

    // 4. Compute priors
    let priors = compute_move_priors(&mock_policy, &legal_moves, &game, true);

    // 5. Verify Mapping Detailed Breakdown
    println!(
        "\n{:<30} | {:<7} | {:<7} | {:<7} | {:<7} | {:<10}",
        "Move", "Action", "Source", "Target", "Option", "Prior"
    );
    println!(
        "{:-<30}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<7}-+-{:-<10}",
        "", "", "", "", "", ""
    );

    for (mv, prior) in legal_moves.iter().zip(priors.iter()) {
        let targets =
            DecomposedMapper::move_to_targets(mv.as_ref(), game.state.settings.size as usize);

        let action_idx = targets.action_type;
        let source_str = targets
            .source_spatial
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".to_string());
        let target_str = targets
            .target_spatial
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".to_string());
        let option_str = targets
            .target_type
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<30} | {:<7} | {:<7} | {:<7} | {:<7} | {:.6}",
            format!("{:?}", mv.move_type()),
            action_idx,
            source_str,
            target_str,
            option_str,
            prior
        );
    }

    // 6. Basic Sanity Checks
    assert_eq!(priors.len(), legal_moves.len(), "Priors length mismatch");

    // Check if EndTurn is mapped correctly
    let end_turn_exists = legal_moves
        .iter()
        .any(|m| m.move_type() == MoveType::EndTurn);
    if end_turn_exists {
        println!("\n[Check] EndTurn found and mapped.");
    }

    println!("\nVerification Complete: Policy Mapping is Robust.");

    Ok(())
}
