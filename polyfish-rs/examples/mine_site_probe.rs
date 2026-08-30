//! Ad-hoc: dump tile/resource/city info for the Build-Mine candidate sites
//! flagged by Verdi at seed0's turn 3 (idx28: 83 vs 85/37/38/39/50) and
//! turn 6 (idx75: 38 vs 50). Read-only.

use polyfish::functions::{get_chebyshev_distance, get_city_at};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};

const REPLAY: &str =
    "replays/exp096_seed0_watch/game_iter1_game0_seed1787500020.replay.json";

fn state_at_step(full: &ModReplay, target_idx: usize) -> Game {
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    return game;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} move not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    panic!("target_idx {target_idx} beyond game length {idx}");
}

fn dump_tile(game: &Game, tile: i32) {
    let size = game.state.settings.size;
    let t = game.state.tiles.get(&tile);
    let terrain = t.map(|t| t.terrain_type);
    let ruling = t.and_then(|t| t.ruling_city_coords);
    let resource = game
        .state
        .resources
        .get(&tile)
        .and_then(|r| r.as_ref())
        .map(|r| r.resource_type);
    let owner = t.map(|t| t.owner).unwrap_or(0);
    let mut cities: Vec<(i32, PlayerIdShim, i32)> = Vec::new();
    for tribe in game.state.tribes.values() {
        for c in &tribe.cities {
            let d = get_chebyshev_distance(tile, c.idx, size);
            cities.push((c.idx, PlayerIdShim(c.owner), d));
        }
    }
    cities.sort_by_key(|&(_, _, d)| d);
    let nearest: Vec<String> = cities
        .iter()
        .take(3)
        .map(|(idx, owner, d)| format!("city{idx}(p{},d{d})", owner.0))
        .collect();
    println!(
        "tile {tile}: terrain={terrain:?} resource={resource:?} owner={owner} ruling_city={ruling:?} nearest_cities=[{}]",
        nearest.join(", ")
    );
}

struct PlayerIdShim(polyfish::PlayerId);

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    println!("=== BEFORE idx28 (turn3 mine build: chose 83) ===");
    let g1 = state_at_step(&full, 28);
    for tile in [83, 85, 37, 38, 39, 50] {
        dump_tile(&g1, tile);
    }
    println!("player1 cities: {:?}", g1.state.tribes.get(&1).map(|t| t.cities.iter().map(|c| c.idx).collect::<Vec<_>>()));

    println!("\n=== BEFORE idx75 (turn6 mine build: chose 38 over 39/50) ===");
    let g2 = state_at_step(&full, 75);
    for tile in [38, 39, 50, 51, 52, 53, 30] {
        dump_tile(&g2, tile);
    }
    println!("player1 cities: {:?}", g2.state.tribes.get(&1).map(|t| t.cities.iter().map(|c| c.idx).collect::<Vec<_>>()));
}
