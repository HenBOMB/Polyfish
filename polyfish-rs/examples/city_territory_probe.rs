//! EXP_ELO_112 verification: per-city territory/level/population snapshot at
//! a given step, so a reward-choice claim (e.g. "BorderGrowth wasn't worth
//! it") can be checked against the actual territory size that fed the
//! CityRewardType::BorderGrowth heuristic, not eyeballed.
//! Usage: cargo run --example city_territory_probe -- <replay.json> <target_idx> <pov>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let target_idx: usize = args[2].parse().unwrap();
    let pov: i32 = args[3].parse().unwrap();
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    'outer: for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    break 'outer;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    let tribe = game.state.tribes.get(&pov).unwrap();
    for c in &tribe.cities {
        println!(
            "city {} territory_len={} level={} pop={}",
            c.idx,
            c._territory.len(),
            c.level,
            c.population
        );
    }
}
