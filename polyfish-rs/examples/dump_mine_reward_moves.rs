//! Walk a replay through the real engine and print every Mine-build and
//! City-Reward move with its running move index, turn, player, and target
//! tile — for correlating a specific decision against eco_plan --explain.
//! Usage: cargo run --example dump_mine_reward_moves -- <replay.json>

use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::types::MoveType;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_mine_reward_moves <replay.json>");
    let data = std::fs::read_to_string(&path).expect("read replay");
    let replay: ModReplay = serde_json::from_str(&data).expect("parse replay");

    let mut game = Game::new();
    game.state = replay.game_state.clone();
    game.post_load();

    let mut idx = 0usize;
    for turn_data in &replay.turns {
        let mut players: Vec<_> = turn_data.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for player_data in players {
            for cmd_json in &player_data.commands {
                let legal = game.legal_moves();
                let matched = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd_json)
                    .unwrap_or_else(|| panic!("idx={idx} move not legal: {cmd_json}"));
                let mt = matched.move_type();
                if matches!(mt, MoveType::Build | MoveType::Harvest | MoveType::Reward) {
                    println!(
                        "idx={idx:4} turn={:3} player={} {:?} target={:?} :: {}",
                        turn_data.turn,
                        player_data.player_id,
                        mt,
                        matched.target_idx().ok(),
                        cmd_json
                    );
                }
                game.play_move(matched.as_ref());
                idx += 1;
            }
        }
    }
}
