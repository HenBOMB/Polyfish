//! EXP_ELO_114: full `combat::city_risks` snapshot for one player at a
//! given step -- sieged/open/reachable/risk/worth/enterers -- so a "should
//! this city be pricing exposure right now" claim can be checked against
//! the exact struct the `city_open_exposed` term (and the older Defend-
//! order cover/hold/recall family) actually reads, not eyeballed.
//! Usage: cargo run --example city_risk_probe -- <replay.json> <target_idx> <pov>
use polyfish::ai::combat::city_risks;
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
    for r in city_risks(&game.state, pov) {
        println!(
            "city={} sieged={} open={} arrives_next_turn={} risk={:.3} at_risk={} worth={:.1} enterers={:?}",
            r.city, r.sieged, r.open, r.arrives_next_turn, r.risk, r.at_risk, r.worth, r.enterers
        );
    }
}
