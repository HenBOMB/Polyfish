//! Ad-hoc: dump the true GameState at an exact replayed move index to JSON,
//! for feeding into `eco_plan --state` (ground-truth economy planner) or
//! other offline tools. Usage: cargo run --example dump_state_at_step --
//! <replay.json> <move_idx> <out.json> <pov>

use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn state_at_step(full: &ModReplay, target_idx: usize) -> Game {
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
                    return game;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} move not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    panic!("target_idx {target_idx} beyond game length {idx}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let replay_path = &args[1];
    let target_idx: usize = args[2].parse().unwrap();
    let out_path = &args[3];
    let pov: i32 = args[4].parse().unwrap();

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let mut game = state_at_step(&full, target_idx);
    game.state.settings.current_player_turn_id = pov;
    std::fs::write(out_path, serde_json::to_string(&game.state).unwrap()).unwrap();
    println!("wrote {out_path} at move_idx={target_idx} pov={pov}");
}
