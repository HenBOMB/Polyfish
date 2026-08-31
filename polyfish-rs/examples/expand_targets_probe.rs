//! EXP_ELO_115: `oracle_macro::expand_targets` output turn-by-turn (real
//! live targets AND belief-guess top-ups, however the current gating logic
//! shapes them), so an "we should've expanded/scouted toward X sooner"
//! claim can be checked against what the Expand-order system actually saw,
//! not just what fog reveals confirm after the fact.
//! Usage: cargo run --example expand_targets_probe -- <replay.json> <pov> <max_turn>
use polyfish::ai::oracle_macro::expand_targets;
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
    for turn in 0..max_turn {
        let g = state_at_turn_start(&full, turn);
        let targets = expand_targets(&g.state, pov, None);
        let cities = g.state.tribes.get(&pov).map_or(0, |t| t.cities.len());
        println!("turn {turn}: cities={cities} expand_targets={targets:?}");
    }
}
