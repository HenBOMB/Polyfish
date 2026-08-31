//! EXP_ELO_115: `belief::prediction::guess_villages` output turn-by-turn,
//! independent of whatever gate `expand_targets` currently applies around
//! it -- lets a "did the belief system actually know about this village
//! early" question be answered directly against the predictor itself.
//! Usage: cargo run --example village_guess_probe -- <replay.json> <pov> <max_turn> <max_sites>
use polyfish::ai::belief::prediction::guess_villages;
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn state_at_turn_start(full: &ModReplay, turn: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    g.state = full.game_state.clone();
    g.post_load();
    'outer: for t in &mr.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                let legal = g.legal_moves();
                let Some(m) = legal.iter().find(|m| &m.serialize() == cmd) else {
                    continue 'outer;
                };
                g.play_move(m.as_ref());
            }
        }
    }
    g
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let pov: i32 = args[2].parse().unwrap();
    let max_turn: i32 = args[3].parse().unwrap();
    let max_sites: usize = args[4].parse().unwrap();
    for turn in 0..max_turn {
        let g = state_at_turn_start(&full, turn);
        let guesses = guess_villages(&g.state, pov, max_sites);
        let tiles: Vec<i32> = guesses.iter().map(|v| v.tile).collect();
        println!("turn {turn}: guessed_villages={tiles:?}");
    }
}
