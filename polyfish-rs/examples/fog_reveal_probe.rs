//! EXP_ELO_112 verification: per-turn fog-reveal status and nearest-own-unit
//! distance for one tile, so a "we could've scouted this sooner" claim can be
//! checked against real vision timing instead of eyeballed from the replay.
//! Usage: cargo run --example fog_reveal_probe -- <replay.json> <pov> <tile>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let pov: i32 = args[2].parse().unwrap();
    let tile: i32 = args[3].parse().unwrap();
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    let mut last_turn = -1;
    let mut first_seen: Option<i32> = None;
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
            let explored = game
                .state
                .tiles
                .get(&tile)
                .map_or(false, |tl| tl.explorers.contains(&pov));
            if explored && first_seen.is_none() {
                first_seen = Some(t.turn);
            }
            let size = game.state.settings.size;
            let (bx, by) = (tile % size, tile / size);
            let mut min_d = i32::MAX;
            if let Some(tr) = game.state.tribes.get(&pov) {
                for u in &tr.units {
                    let (ax, ay) = (u.coords.idx % size, u.coords.idx / size);
                    let d = (ax - bx).abs().max((ay - by).abs());
                    min_d = min_d.min(d);
                }
            }
            println!(
                "turn {}: tile{tile} explored_by_p{pov}={explored} nearest_own_unit_dist={min_d}",
                t.turn
            );
        }
    }
    println!("first_seen={first_seen:?}");
}
