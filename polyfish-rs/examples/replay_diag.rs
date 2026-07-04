//! Temporary diagnostic: replay a self-play recap through the real engine,
//! printing the legal-move set at every decision point. Detects (a) decisions
//! where EndTurn was truly the only legal move vs (b) recorded moves that are
//! not legal / fail to execute.

use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::types::MoveType;
use std::collections::BTreeMap;

fn type_counts(moves: &[Box<dyn polyfish::moves::Move>]) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for mv in moves {
        *m.entry(format!("{:?}", mv.move_type())).or_insert(0) += 1;
    }
    m
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: replay_diag <recap.json>");
    let data = std::fs::read_to_string(&path).expect("read recap");
    let replay: ModReplay = serde_json::from_str(&data).expect("parse recap");

    let mut game = Game::new();
    game.state = replay.game_state.clone();
    game.post_load();

    for turn_data in &replay.turns {
        for player_data in &turn_data.players {
            for cmd_json in &player_data.commands {
                let legal = game.legal_moves();
                let cur_player = game.state.settings.current_player_turn_id;
                let stars = game
                    .state
                    .tribes
                    .get(&cur_player)
                    .map(|t| t.stars)
                    .unwrap_or(-1);

                let cmd_type = cmd_json
                    .get("moveType")
                    .and_then(|v| v.as_i64())
                    .map(|v| format!("{:?}", MoveType::from(v as i32)))
                    .unwrap_or_else(|| "?".into());

                println!(
                    "turn {:2} rec_p{} eng_p{} stars {:3} | legal {:3} {:?} | recorded: {} {}",
                    turn_data.turn,
                    player_data.player_id,
                    cur_player,
                    stars,
                    legal.len(),
                    type_counts(&legal),
                    cmd_type,
                    cmd_json
                );

                // Find the recorded move among legal moves by serialization.
                let matched = legal.iter().find(|m| &m.serialize() == cmd_json);
                match matched {
                    Some(m) => {
                        if game.play_move(m.as_ref()).is_none() {
                            println!("  !! EXEC FAILED for recorded move");
                        }
                    }
                    None => {
                        println!("  !! RECORDED MOVE NOT IN LEGAL SET");
                    }
                }
            }
        }
    }
    println!("done. final turn {}", game.state.settings.turn);
}
