//! One-off: is the unit garrisoning city41 at turn 12 the SAME unit that
//! vacated at turn 11, or a different one? Also prints unit_worth for every
//! player-1 unit within distance 2 of tile41 at the moment of the vacate, to
//! check whether cheaper eligible units existed.
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let city: i32 = 41;
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
            let garrison = polyfish::functions::get_unit_at(&game.state, city)
                .map(|u| (u.id, u.owner, u.health, format!("{:?}", u.unit_type)));
            println!("turn {}: city{city} garrison_unit={:?}", t.turn, garrison);
            if t.turn == 11 {
                let mut nearby: Vec<(u32, i32, String, f32, f32)> = Vec::new();
                if let Some(tribe) = game.state.tribes.get(&1) {
                    for u in &tribe.units {
                        let size = game.state.settings.size;
                        let (ax, ay) = (u.coords.idx % size, u.coords.idx / size);
                        let (bx, by) = (city % size, city / size);
                        let d = (ax - bx).abs().max((ay - by).abs());
                        if d <= 2 {
                            nearby.push((
                                u.id,
                                d,
                                format!("{:?}", u.unit_type),
                                u.health,
                                polyfish::rules::combat::unit_worth(u) as f32,
                            ));
                        }
                    }
                }
                nearby.sort_by_key(|x| x.1);
                println!("  turn11 nearby player1 units (id, dist, type, hp, worth): {nearby:?}");
            }
        }
    }
}
