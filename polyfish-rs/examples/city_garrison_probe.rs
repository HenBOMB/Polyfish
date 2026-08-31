//! Generalized city49_probe.rs: per-turn garrison + nearby-enemy snapshot for
//! any city tile, so undefended-window claims can be checked against real
//! enemy proximity instead of eyeballed from the console log.
//! Usage: cargo run --example city_garrison_probe -- <replay.json> <city_idx>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let city: i32 = args[2].parse().unwrap();
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
            let owner = game.state.tiles.get(&city).map(|tl| tl.owner);
            let garrison = polyfish::functions::get_unit_at(&game.state, city)
                .map(|u| (u.owner, u.health, format!("{:?}", u.unit_type)));
            let mut nearby: Vec<(i32, i32, i32, f32)> = Vec::new();
            for (pid, tribe) in &game.state.tribes {
                for u in &tribe.units {
                    let d = {
                        let size = game.state.settings.size;
                        let (ax, ay) = (u.coords.idx % size, u.coords.idx / size);
                        let (bx, by) = (city % size, city / size);
                        (ax - bx).abs().max((ay - by).abs())
                    };
                    if d <= 3 {
                        nearby.push((*pid, u.coords.idx, d, u.health));
                    }
                }
            }
            nearby.sort_by_key(|x| (x.0, x.2));
            println!(
                "turn {}: tile{city} owner={:?} garrison={:?} nearby={:?}",
                t.turn, owner, garrison, nearby
            );
        }
    }
}
