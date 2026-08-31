//! Per-turn unit-type counts for one player, plus star bank and city count,
//! so "N warriors is wasteful" / "giants came too late" claims can be checked
//! against real numbers instead of an eyeballed unit tally.
//! Usage: cargo run --example army_composition_probe -- <replay.json> <pov>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let pov: i32 = args[2].parse().unwrap();
    let raw = std::fs::read_to_string(path).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    let mut last_turn = -1;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
        if t.turn != last_turn {
            last_turn = t.turn;
            let Some(tribe) = game.state.tribes.get(&pov) else {
                continue;
            };
            let mut counts: BTreeMap<String, u32> = BTreeMap::new();
            for u in &tribe.units {
                *counts.entry(format!("{:?}", u.unit_type)).or_insert(0) += 1;
            }
            let stars = tribe.stars;
            let cities = tribe.cities.len();
            println!(
                "turn {}: stars={stars} cities={cities} units={counts:?}",
                t.turn
            );
        }
    }
}
