//! One-off lookup: given a replay + ply_trace.jsonl + a target turn/player and
//! a move signature (moveType/src/target), find every global command index
//! whose reconstructed state has that move as a legal candidate AND whose
//! matching ply_trace row (same turn/player) lists it with a score — printing
//! (target_idx, trace_line_idx, score) triples for attack_pricing_probe3.
use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::replayer::{replay_game, ModReplay};

fn state_at_step(full: &ModReplay, target_idx: usize) -> Option<(Game, i32, i32)> {
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    let turn = game.state.settings.turn;
                    return Some((game, turn, pl.player_id));
                }
                let legal = game.legal_moves();
                let m = legal.iter().find(|m| &m.serialize() == cmd)?;
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let replay_path = &args[1];
    let trace_path = &args[2];
    let target_turn: i32 = args[3].parse().unwrap();
    let pov: i32 = args[4].parse().unwrap();
    let move_type: i64 = args[5].parse().unwrap();
    let src: i64 = args[6].parse().unwrap();
    let target: i64 = args[7].parse().unwrap();
    let lo: usize = args[8].parse().unwrap();
    let hi: usize = args[9].parse().unwrap();

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    let trace_raw = std::fs::read_to_string(trace_path).expect("read ply_trace");
    let trace_rows: Vec<serde_json::Value> = trace_raw
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    for target_idx in lo..hi {
        let Some((mut game, turn, player)) = state_at_step(&full, target_idx) else { continue };
        if turn != target_turn || player != pov {
            continue;
        }
        let legal = game.legal_moves();
        let has_move = legal.iter().any(|m| {
            let s = m.serialize();
            s.get("moveType").and_then(|v| v.as_i64()) == Some(move_type)
                && s.get("src").and_then(|v| v.as_i64()) == Some(src)
                && s.get("target").and_then(|v| v.as_i64()) == Some(target)
        });
        if !has_move {
            continue;
        }
        // find matching trace row: turn/player match AND candidates contains this move with a score
        for (li, row) in trace_rows.iter().enumerate() {
            if row["turn"].as_i64() != Some(target_turn as i64) {
                continue;
            }
            if row["player"].as_i64() != Some(pov as i64) {
                continue;
            }
            if let Some(cands) = row["candidates"].as_array() {
                for c in cands {
                    let m = &c["move"];
                    if m.get("moveType").and_then(|v| v.as_i64()) == Some(move_type)
                        && m.get("src").and_then(|v| v.as_i64()) == Some(src)
                        && m.get("target").and_then(|v| v.as_i64()) == Some(target)
                    {
                        println!(
                            "target_idx={target_idx} trace_line_idx={li} score={}",
                            c["score"]
                        );
                    }
                }
            }
        }
        let _ = game.state.settings.turn; // silence unused warnings in release-ish builds
    }
}
