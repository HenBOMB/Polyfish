//! EXP_ELO_110: which of pov's own unit IDs are lost over a game, and at
//! what turn -- for causally checking whether specific flagged deaths
//! (id14 t9, id16 t12) were actually prevented, not just eyeballing the
//! aggregate units_lost count.
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let pov: i32 = args[2].parse().unwrap();

    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut alive: HashSet<u32> = game
        .state
        .tribes
        .get(&pov)
        .map(|t| t.units.iter().map(|u| u.id).collect())
        .unwrap_or_default();

    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                let legal = game.legal_moves();
                if let Some(m) = legal.iter().find(|m| &m.serialize() == cmd) {
                    game.play_move(m.as_ref());
                }
            }
        }
        let now: HashSet<u32> = game
            .state
            .tribes
            .get(&pov)
            .map(|tr| tr.units.iter().map(|u| u.id).collect())
            .unwrap_or_default();
        for id in alive.difference(&now) {
            println!("lost unit_id={id} by end of turn={}", t.turn);
        }
        alive = now;
    }
}
