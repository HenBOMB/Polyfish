use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::replayer::{replay_game, ModReplay};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let raw = std::fs::read_to_string(&path).unwrap();
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
                let m = legal.iter().find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
        if t.turn != last_turn {
            last_turn = t.turn;
            let owner = game.state.tiles.get(&49).map(|tl| tl.owner);
            let garrison = polyfish::functions::get_unit_at(&game.state, 49)
                .map(|u| (u.owner, u.health, format!("{:?}", u.unit_type)));
            // nearby enemy units (within 3 tiles of 49)
            let mut nearby: Vec<(i32,i32,i32,f32)> = Vec::new();
            for (pid, tribe) in &game.state.tribes {
                for u in &tribe.units {
                    let d = {
                        let (ax,ay) = (u.coords.idx % 11, u.coords.idx / 11);
                        let (bx,by) = (49 % 11, 49 / 11);
                        (ax-bx).abs().max((ay-by).abs())
                    };
                    if d <= 3 {
                        nearby.push((*pid, u.coords.idx, d, u.health));
                    }
                }
            }
            nearby.sort_by_key(|x| (x.0, x.2));
            println!("turn {}: tile49 owner={:?} garrison={:?} nearby={:?}", t.turn, owner, garrison, nearby);
        }
    }
}
