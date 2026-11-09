//! Neural Behavioral Probe: Zero-Search Tactical Interpretability Test
//!
//! Evaluates whether the trained neural network weights directly internalized tactical
//! primitives into their raw policy priors pi_theta(a|s) and value predictions V(s)
//! WITHOUT Monte Carlo Tree Search (0-ply inference).
//!
//! Compares:
//! - Checkpoint Iter 76: `model_dagger_foundation.safetensors` (Campaign Baseline)
//! - Checkpoint Iter 129: `model_iter129.safetensors` (Latest Adversarial Self-Play)

use candle_core::Device;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::features::state_to_cpu_features;
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::ai::policy_composer::compute_move_priors_raw;
use polyfish::coords::Coords;
use polyfish::functions::get_adjacent_indices;
use polyfish::game::Game;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::moves::attack::AttackMove;
use polyfish::moves::step::StepMove;
use polyfish::moves::Move;
use polyfish::types::{MapSize, MapType, MoveType, TerrainType, TribeType, UnitType};
use std::sync::Arc;

fn load_network(path: &str) -> Arc<PolyZeroNet> {
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &Device::Cpu)
            .unwrap_or_else(|e| panic!("failed to load weights from {path}: {e}"))
    };
    Arc::new(PolyZeroNet::new(vs).unwrap_or_else(|e| panic!("failed to initialize PolyZeroNet from {path}: {e}")))
}

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();
    game
}

/// Evaluates raw 0-search policy priors and value for the current board state.
fn evaluate_zero_search(
    net: &Arc<PolyZeroNet>,
    game: &Game,
    pov: i32,
) -> (f32, Vec<(Box<dyn Move>, f32)>) {
    let handle = InlineEvalHandle::new(Arc::clone(net));
    let feat = state_to_cpu_features(&game.state, pov).expect("feature extraction");
    let results = handle.evaluate(vec![feat]);
    let (val_win, _val_prog, raw_policy) = &results[0];

    let legal_moves = game.legal_moves();
    let size = game.state.settings.size as usize;
    let priors = compute_move_priors_raw(raw_policy, &legal_moves, size, true);

    let paired: Vec<(Box<dyn Move>, f32)> = legal_moves
        .into_iter()
        .zip(priors.into_iter())
        .collect();

    (*val_win, paired)
}

#[test]
fn probe_1_rider_hit_and_run_escape() {
    println!("\n=======================================================");
    println!("PROBE 1: Rider Hit-and-Run Escape Prior (0-Search)");
    println!("=======================================================");

    let mut game = make_game(42);
    let size = game.state.settings.size;

    // Find 3 land tiles in line: t_retreat <-> t_p1 <-> t_p2
    let (t_retreat, t_p1, t_p2) = (0..size * size)
        .find_map(|p1| {
            if !matches!(
                game.state.tiles[&p1].terrain_type,
                TerrainType::Field | TerrainType::Forest
            ) {
                return None;
            }
            let adj = polyfish::functions::get_adjacent_indices(&game.state, p1, 1);
            for &p2 in &adj {
                if !matches!(
                    game.state.tiles[&p2].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                ) {
                    continue;
                }
                for &ret in &adj {
                    if ret != p2
                        && matches!(
                            game.state.tiles[&ret].terrain_type,
                            TerrainType::Field | TerrainType::Forest
                        )
                    {
                        let d = polyfish::functions::get_chebyshev_distance(p2, ret, size);
                        if d == 2 {
                            return Some((ret, p1, p2));
                        }
                    }
                }
            }
            None
        })
        .expect("must find retreat line of 3 traversable land tiles");

    for &idx in &[t_retreat, t_p1, t_p2] {
        let tile = game.state.tiles.get_mut(&idx).unwrap();
        tile.explorers.insert(1);
        tile.explorers.insert(2);
    }
    for tile in game.state.tiles.values_mut() {
        tile._unit_owner_id = None;
    }

    // Set up P1 Rider at t_p1 and P2 Warrior at t_p2
    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        p1.units.truncate(1);
        p1.units[0].unit_type = UnitType::Rider;
        p1.units[0].health = 10.0;
        p1.units[0].attacked = false;
        p1.units[0].moved = false;
        p1.units[0].coords = Coords::from_index(t_p1, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        p2.units.truncate(1);
        p2.units[0].unit_type = UnitType::Warrior;
        p2.units[0].health = 10.0;
        p2.units[0].attacked = false;
        p2.units[0].moved = false;
        p2.units[0].coords = Coords::from_index(t_p2, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    game.post_load();

    // Rider executes attack on enemy Warrior
    let atk = AttackMove::new(t_p1, t_p2);
    let _ = game.simulate_move(&atk).expect("Rider attack simulates");

    println!("Geometry: Rider at tile {} ({:?}), Enemy Warrior at tile {} ({:?})",
        t_p1, Coords::from_index(t_p1, size), t_p2, Coords::from_index(t_p2, size));
    println!("Candidate retreat tile: {} ({:?}) (Chebyshev dist to enemy: {})",
        t_retreat, Coords::from_index(t_retreat, size),
        polyfish::functions::get_chebyshev_distance(t_retreat, t_p2, size));
    println!("After attack: Rider HP = {:.1}, Enemy Warrior HP = {:.1}",
        game.state.tribes[&1].units[0].health,
        game.state.tribes[&2].units[0].health);
    for idx in &[3, 12, 13, 14] {
        let t = &game.state.tiles[idx];
        let res = game.state.resources.get(idx).and_then(|r| r.as_ref()).map(|r| r.resource_type);
        println!("Tile {:>2}: terrain={:?}, resource={:?}, owner={:?}",
            idx, t.terrain_type, res, t.owner);
    }

    // Verify Rider has attacked=true, moved=false (Escape skill active)
    assert!(game.state.tribes[&1].units[0].attacked);
    assert!(!game.state.tribes[&1].units[0].moved);

    // Now test raw network opinions on:
    // 1. StepMove to t_retreat (the escape move)
    // 2. EndTurnMove (stay in fatal danger)
    let net_76 = load_network("model_dagger_foundation.safetensors");
    let net_129 = load_network("model_iter129.safetensors");
    let net_exp002_99 = if std::path::Path::new("model_exp002_iter99.safetensors").exists() {
        Some(load_network("model_exp002_iter99.safetensors"))
    } else {
        None
    };
    let net_exp002_129 = if std::path::Path::new("model_exp002_iter129.safetensors").exists() {
        Some(load_network("model_exp002_iter129.safetensors"))
    } else {
        None
    };

    let (val_76, moves_76) = evaluate_zero_search(&net_76, &game, 1);
    let (val_129, moves_129) = evaluate_zero_search(&net_129, &game, 1);
    let res_exp002_99 = net_exp002_99.as_ref().map(|n| evaluate_zero_search(n, &game, 1));
    let res_exp002_129 = net_exp002_129.as_ref().map(|n| evaluate_zero_search(n, &game, 1));

    let get_prior = |moves: &[(Box<dyn Move>, f32)], m_type: MoveType, target: Option<usize>| -> f32 {
        moves
            .iter()
            .find(|(m, _)| {
                m.move_type() == m_type && (target.is_none() || m.target_idx().ok() == target)
            })
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    };

    let p_esc_76 = get_prior(&moves_76, MoveType::Step, Some(t_retreat as usize));
    let p_end_76 = get_prior(&moves_76, MoveType::EndTurn, None);
    let p_esc_129 = get_prior(&moves_129, MoveType::Step, Some(t_retreat as usize));
    let p_end_129 = get_prior(&moves_129, MoveType::EndTurn, None);

    let ratio_76 = if p_end_76 > 0.0 { p_esc_76 / p_end_76 } else { 0.0 };
    let ratio_129 = if p_end_129 > 0.0 { p_esc_129 / p_end_129 } else { 0.0 };

    println!("Model Iter 76:  Escape Prior: {:.6} | EndTurn Prior: {:.6} | Ratio (Escape/End): {:.3} | Value: {:.3}",
        p_esc_76, p_end_76, ratio_76, val_76);
    println!("Model Iter 129: Escape Prior: {:.6} | EndTurn Prior: {:.6} | Ratio (Escape/End): {:.3} | Value: {:.3}",
        p_esc_129, p_end_129, ratio_129, val_129);

    if let Some((val_exp002, ref moves_exp002)) = res_exp002_99 {
        let p_esc_exp002 = get_prior(moves_exp002, MoveType::Step, Some(t_retreat as usize));
        let p_end_exp002 = get_prior(moves_exp002, MoveType::EndTurn, None);
        let ratio_exp002 = if p_end_exp002 > 0.0 { p_esc_exp002 / p_end_exp002 } else { 0.0 };
        println!("EXP_002 Iter 99:  Escape Prior: {:.6} | EndTurn Prior: {:.6} | Ratio (Escape/End): {:.3} | Value: {:.3}",
            p_esc_exp002, p_end_exp002, ratio_exp002, val_exp002);
    }
    if let Some((val_exp002, ref moves_exp002)) = res_exp002_129 {
        let p_esc_exp002 = get_prior(moves_exp002, MoveType::Step, Some(t_retreat as usize));
        let p_end_exp002 = get_prior(moves_exp002, MoveType::EndTurn, None);
        let ratio_exp002 = if p_end_exp002 > 0.0 { p_esc_exp002 / p_end_exp002 } else { 0.0 };
        println!("EXP_002 Iter 129: Escape Prior: {:.6} | EndTurn Prior: {:.6} | Ratio (Escape/End): {:.3} | Value: {:.3}",
            p_esc_exp002, p_end_exp002, ratio_exp002, val_exp002);
    }

    println!("Relative escape preference change: {:+.1}%", ((ratio_129 / ratio_76) - 1.0) * 100.0);

    println!("\nTop 5 Legal Moves by Prior for Iter 76:");
    let mut indices_76: Vec<usize> = (0..moves_76.len()).collect();
    indices_76.sort_by(|&a, &b| moves_76[b].1.partial_cmp(&moves_76[a].1).unwrap());
    for (i, &idx) in indices_76.iter().take(5).enumerate() {
        let (ref m, p) = moves_76[idx];
        let detail = if let Ok(t) = m.target_idx() {
            let dist = polyfish::functions::get_chebyshev_distance(t as i32, t_p2, size);
            format!("target={} ({:?}), dist_to_enemy={}, safe={}", t, Coords::from_index(t as i32, size), dist, dist > 1)
        } else {
            "non-spatial".to_string()
        };
        println!("  {}. Move: {:?} | Prior: {:.6} | {}", i + 1, m, p, detail);
    }

    println!("\nTop 5 Legal Moves by Prior for Iter 129:");
    let mut indices_129: Vec<usize> = (0..moves_129.len()).collect();
    indices_129.sort_by(|&a, &b| moves_129[b].1.partial_cmp(&moves_129[a].1).unwrap());
    for (i, &idx) in indices_129.iter().take(5).enumerate() {
        let (ref m, p) = moves_129[idx];
        let detail = if let Ok(t) = m.target_idx() {
            let dist = polyfish::functions::get_chebyshev_distance(t as i32, t_p2, size);
            format!("target={} ({:?}), dist_to_enemy={}, safe={}", t, Coords::from_index(t as i32, size), dist, dist > 1)
        } else {
            "non-spatial".to_string()
        };
        println!("  {}. Move: {:?} | Prior: {:.6} | {}", i + 1, m, p, detail);
    }

    println!("\n--- SIMULATION 1: Raw 0-Search Argmax Execution ---");
    let top_76 = &moves_76[indices_76[0]].0;
    let mut g_sim_0_76 = game.clone();
    let _ = g_sim_0_76.simulate_move(top_76.as_ref());
    let r_pos_76 = g_sim_0_76.state.tribes[&1].units[0].coords.idx;
    let d_76 = polyfish::functions::get_chebyshev_distance(r_pos_76, t_p2, size);
    println!("Iter 76 0-Search chose: {:?} -> Rider at tile {} (dist to enemy: {}, safe: {})",
        top_76, r_pos_76, d_76, d_76 > 1);

    let top_129 = &moves_129[indices_129[0]].0;
    let mut g_sim_0_129 = game.clone();
    let _ = g_sim_0_129.simulate_move(top_129.as_ref());
    let r_pos_129 = g_sim_0_129.state.tribes[&1].units[0].coords.idx;
    let d_129 = polyfish::functions::get_chebyshev_distance(r_pos_129, t_p2, size);
    println!("Iter 129 0-Search chose: {:?} -> Rider at tile {} (dist to enemy: {}, safe: {})",
        top_129, r_pos_129, d_129, d_129 > 1);

    println!("\n--- SIMULATION 2: Full Gumbel MCTS Search Execution (128 visits, k=16) ---");
    let eval_76 = Evaluator::Inline(InlineEvalHandle::new(Arc::clone(&net_76)));
    let mut mcts_76 = GumbelMctsAgent::new(&eval_76, 128, 16).with_search_seed(42);
    let mut g_mcts_76 = game.clone();
    let mcts_move_76 = mcts_76.select_move(&mut g_mcts_76).expect("MCTS 76 selects move");
    let mut g_res_76 = game.clone();
    let _ = g_res_76.simulate_move(mcts_move_76.as_ref());
    let mcts_pos_76 = g_res_76.state.tribes[&1].units[0].coords.idx;
    let mcts_d_76 = polyfish::functions::get_chebyshev_distance(mcts_pos_76, t_p2, size);
    println!("Iter 76 MCTS chose: {:?} -> Rider at tile {} (dist to enemy: {}, safe: {})",
        mcts_move_76, mcts_pos_76, mcts_d_76, mcts_d_76 > 1);

    let eval_129 = Evaluator::Inline(InlineEvalHandle::new(Arc::clone(&net_129)));
    let mut mcts_129 = GumbelMctsAgent::new(&eval_129, 128, 16).with_search_seed(42);
    mcts_129.arm_trace();
    let mut g_mcts_129 = game.clone();
    let (mcts_move_129_opt, _) = mcts_129.select_move_with_stats(&mut g_mcts_129);
    let mcts_move_129 = mcts_move_129_opt.expect("MCTS 129 selects move");
    let trace_129 = mcts_129.take_trace().expect("trace 129");
    let mut g_res_129 = game.clone();
    let _ = g_res_129.simulate_move(mcts_move_129.as_ref());
    let mcts_pos_129 = g_res_129.state.tribes[&1].units[0].coords.idx;
    let mcts_d_129 = polyfish::functions::get_chebyshev_distance(mcts_pos_129, t_p2, size);
    println!("Iter 129 MCTS chose: {:?} -> Rider at tile {} (dist to enemy: {}, safe: {})",
        mcts_move_129, mcts_pos_129, mcts_d_129, mcts_d_129 > 1);

    println!("\nMCTS Search Root Candidates for Iter 129:");
    for cand in &trace_129.candidates {
        let dist = cand.target_idx.map(|t| polyfish::functions::get_chebyshev_distance(t as i32, t_p2, size));
        println!("  Candidate: {:<35} | Visits: {:>3.0} | Q: {:+.4} | OwnVal: {:?} | Prior: {:.5} | DistToEnemy: {:?}",
            cand.description, cand.visits, cand.q_value, cand.own_value, cand.search_prior_prob, dist);
    }
}

#[test]
fn probe_2_fatal_suicide_attack_discouragement() {
    println!("\n=======================================================");
    println!("PROBE 2: Fatal Suicide Attack Prior (0-Search)");
    println!("=======================================================");

    let mut game = make_game(42);
    let size = game.state.settings.size;

    let t_p1 = 12;
    let neighbors = get_adjacent_indices(&game.state, t_p1, 1);
    let t_p2 = neighbors[0];

    // P1 Warrior at 3.0 HP vs P2 Defender at 15.0 HP
    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        p1.units.truncate(1);
        p1.units[0].unit_type = UnitType::Warrior;
        p1.units[0].health = 3.0;
        p1.units[0].attacked = false;
        p1.units[0].moved = false;
        p1.units[0].coords = Coords::from_index(t_p1, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        p2.units.truncate(1);
        p2.units[0].unit_type = UnitType::Defender;
        p2.units[0].health = 15.0;
        p2.units[0].attacked = false;
        p2.units[0].moved = false;
        p2.units[0].coords = Coords::from_index(t_p2, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    game.post_load();

    let net_76 = load_network("model_dagger_foundation.safetensors");
    let net_129 = load_network("model_iter129.safetensors");
    let net_exp002_99 = if std::path::Path::new("model_exp002_iter99.safetensors").exists() {
        Some(load_network("model_exp002_iter99.safetensors"))
    } else {
        None
    };
    let net_exp002_129 = if std::path::Path::new("model_exp002_iter129.safetensors").exists() {
        Some(load_network("model_exp002_iter129.safetensors"))
    } else {
        None
    };

    let (val_76, moves_76) = evaluate_zero_search(&net_76, &game, 1);
    let (val_129, moves_129) = evaluate_zero_search(&net_129, &game, 1);
    let res_exp002_99 = net_exp002_99.as_ref().map(|n| evaluate_zero_search(n, &game, 1));
    let res_exp002_129 = net_exp002_129.as_ref().map(|n| evaluate_zero_search(n, &game, 1));

    let get_prior = |moves: &[(Box<dyn Move>, f32)], m_type: MoveType, target: Option<usize>| -> f32 {
        moves
            .iter()
            .find(|(m, _)| {
                m.move_type() == m_type && (target.is_none() || m.target_idx().ok() == target)
            })
            .map(|(_, p)| *p)
            .unwrap_or(0.0)
    };

    let p_suicide_76 = get_prior(&moves_76, MoveType::Attack, Some(t_p2 as usize));
    let p_end_76 = get_prior(&moves_76, MoveType::EndTurn, None);
    let p_suicide_129 = get_prior(&moves_129, MoveType::Attack, Some(t_p2 as usize));
    let p_end_129 = get_prior(&moves_129, MoveType::EndTurn, None);

    println!("Model Iter 76:  Suicide Prior: {:.6} | EndTurn Prior: {:.6} | Value: {:.3}",
        p_suicide_76, p_end_76, val_76);
    println!("Model Iter 129: Suicide Prior: {:.6} | EndTurn Prior: {:.6} | Value: {:.3}",
        p_suicide_129, p_end_129, val_129);

    if let Some((val_exp002, ref moves_exp002)) = res_exp002_99 {
        let p_suicide_exp002 = get_prior(moves_exp002, MoveType::Attack, Some(t_p2 as usize));
        let p_end_exp002 = get_prior(moves_exp002, MoveType::EndTurn, None);
        println!("EXP_002 Iter 99:  Suicide Prior: {:.6} | EndTurn Prior: {:.6} | Value: {:.3}",
            p_suicide_exp002, p_end_exp002, val_exp002);
    }
    if let Some((val_exp002, ref moves_exp002)) = res_exp002_129 {
        let p_suicide_exp002 = get_prior(moves_exp002, MoveType::Attack, Some(t_p2 as usize));
        let p_end_exp002 = get_prior(moves_exp002, MoveType::EndTurn, None);
        println!("EXP_002 Iter 129: Suicide Prior: {:.6} | EndTurn Prior: {:.6} | Value: {:.3}",
            p_suicide_exp002, p_end_exp002, val_exp002);
    }

    println!("Suicide attack probability delta: {:+.2}%", (p_suicide_129 - p_suicide_76) * 100.0);
}

#[test]
fn probe_3_capital_defense_abandonment() {
    println!("\n=======================================================");
    println!("PROBE 3: Capital Abandonment & Defense Value Penalty");
    println!("=======================================================");

    let mut game = make_game(7);
    let size = game.state.settings.size;
    let city_idx = game.state.tribes[&1].cities[0].idx;

    let neighbors = get_adjacent_indices(&game.state, city_idx, 1);
    let t_enemy = neighbors[0];
    let t_empty = neighbors[1];

    // Set up P1 Defender on the Capital (city_idx)
    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        p1.units.truncate(1);
        p1.units[0].unit_type = UnitType::Defender;
        p1.units[0].health = 15.0;
        p1.units[0].attacked = false;
        p1.units[0].moved = false;
        p1.units[0].coords = Coords::from_index(city_idx, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    // Set up P2 Warrior threatening the Capital from t_enemy
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        p2.units.truncate(1);
        p2.units[0].unit_type = UnitType::Warrior;
        p2.units[0].health = 10.0;
        p2.units[0].attacked = false;
        p2.units[0].moved = false;
        p2.units[0].coords = Coords::from_index(t_enemy, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    game.post_load();

    let net_76 = load_network("model_dagger_foundation.safetensors");
    let net_129 = load_network("model_iter129.safetensors");
    let net_exp002_99 = if std::path::Path::new("model_exp002_iter99.safetensors").exists() {
        Some(load_network("model_exp002_iter99.safetensors"))
    } else {
        None
    };
    let net_exp002_129 = if std::path::Path::new("model_exp002_iter129.safetensors").exists() {
        Some(load_network("model_exp002_iter129.safetensors"))
    } else {
        None
    };

    // Case A: Defender is GARRISONED on Capital
    let (val_garrison_76, _) = evaluate_zero_search(&net_76, &game, 1);
    let (val_garrison_129, _) = evaluate_zero_search(&net_129, &game, 1);
    let val_garrison_exp002_99 = net_exp002_99.as_ref().map(|n| evaluate_zero_search(n, &game, 1).0);
    let val_garrison_exp002_129 = net_exp002_129.as_ref().map(|n| evaluate_zero_search(n, &game, 1).0);

    // Case B: Defender STEPS OFF the capital to t_empty (leaving capital bare)
    let mut game_abandoned = game.clone();
    let step_away = StepMove::new(city_idx, t_empty);
    let _ = game_abandoned.simulate_move(&step_away);

    let (val_abandoned_76, _) = evaluate_zero_search(&net_76, &game_abandoned, 1);
    let (val_abandoned_129, _) = evaluate_zero_search(&net_129, &game_abandoned, 1);
    let val_abandoned_exp002_99 = net_exp002_99.as_ref().map(|n| evaluate_zero_search(n, &game_abandoned, 1).0);
    let val_abandoned_exp002_129 = net_exp002_129.as_ref().map(|n| evaluate_zero_search(n, &game_abandoned, 1).0);

    let penalty_76 = val_abandoned_76 - val_garrison_76;
    let penalty_129 = val_abandoned_129 - val_garrison_129;

    println!("Model Iter 76:  Garrison Val: {:+.3} | Abandoned Val: {:+.3} | Delta: {:+.3}",
        val_garrison_76, val_abandoned_76, penalty_76);
    println!("Model Iter 129: Garrison Val: {:+.3} | Abandoned Val: {:+.3} | Delta: {:+.3}",
        val_garrison_129, val_abandoned_129, penalty_129);

    if let (Some(g_val), Some(a_val)) = (val_garrison_exp002_99, val_abandoned_exp002_99) {
        let penalty_exp002 = a_val - g_val;
        println!("EXP_002 Iter 99:  Garrison Val: {:+.3} | Abandoned Val: {:+.3} | Delta: {:+.3}",
            g_val, a_val, penalty_exp002);
    }
    if let (Some(g_val), Some(a_val)) = (val_garrison_exp002_129, val_abandoned_exp002_129) {
        let penalty_exp002 = a_val - g_val;
        println!("EXP_002 Iter 129: Garrison Val: {:+.3} | Abandoned Val: {:+.3} | Delta: {:+.3}",
            g_val, a_val, penalty_exp002);
    }

    println!("Capital defense sensitivity multiplier: {:.2}x", penalty_129 / penalty_76);
}

