//! Full move log for a replay, one line per move, for manual whole-turn
//! review. Usage: cargo run --example dump_all_moves -- <replay.json> [player]
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let only_player: Option<i32> = args.get(2).and_then(|s| s.parse().ok());

    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();

    let mut idx = 0usize;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            if let Some(p) = only_player {
                if pl.player_id != p {
                    for cmd in &pl.commands {
                        let legal = game.legal_moves();
                        let m = legal.iter().find(|m| &m.serialize() == cmd)
                            .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                        game.play_move(m.as_ref());
                        idx += 1;
                    }
                    continue;
                }
            }
            for cmd in &pl.commands {
                let legal = game.legal_moves();
                let m = legal.iter().find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                println!("idx={idx:4} turn={:3} p{} {:?} :: {}", t.turn, pl.player_id, m.move_type(), m.serialize());
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
}
