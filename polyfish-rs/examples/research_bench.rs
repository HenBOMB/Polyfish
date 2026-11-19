use std::{hint::black_box, time::Instant};

use anyhow::{Context, Result};
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::EndTurnMove;
use polyfish::research::{self, MatchConfig, Opponent};
use polyfish::types::{TerrainType, TribeType};
use serde_json::json;

fn flat_board(tiles: u16) -> Result<Game> {
    let map_size = research::board_size(tiles).context("Unsupported board size")?;
    let mut game = research::new_match(
        map_size,
        [TribeType::XinXi, TribeType::Imperius],
        42,
        MatchConfig::new(Opponent::Human, 2),
    );
    let size = game.map_size();
    game.state.structures.clear();
    game.state.resources.clear();
    for tile in game.state.tiles.values_mut() {
        *tile = polyfish::states::TileState {
            coords: tile.coords,
            terrain_type: TerrainType::Field,
            ..Default::default()
        };
    }
    for (seat, pos) in [(1, 1), (2, size - 2)] {
        let index = pos * size + pos;
        let coords = Coords::from_index(index, size);
        let tribe = &mut game.state.tribes[&seat];
        let city = &mut tribe.cities[0];
        city.idx = index;
        city._territory.clear();
        tribe.starting_tile_coords = coords;
        tribe.units[0].coords = coords;
        tribe.units[0].prev_coords = coords;
        tribe.units[0].home_coords = Some(coords);
        for tile in game.state.tiles.values_mut() {
            if polyfish::functions::get_chebyshev_distance(index, tile.coords.idx, size) <= 1 {
                tile.owner = seat;
                tile.ruling_city_coords = Some(coords);
                city._territory.push(tile.coords.idx);
            }
        }
        game.state.tiles[&index].terrain_type = TerrainType::Field;
        game.state.tiles[&index].capital_of = seat;
    }
    game.post_load();
    Ok(game)
}

fn run(game: &Game, iterations: usize) -> [u128; 4] {
    let mut times = [0; 4];
    for _ in 0..iterations {
        let start = Instant::now();
        let copy = black_box(game.clone());
        times[0] += start.elapsed().as_nanos();
        drop(copy);
        let start = Instant::now();
        let view = black_box(research::observation(game, 1));
        times[1] += start.elapsed().as_nanos();
        drop(view);
        let mut next = game.clone();
        let start = Instant::now();
        let legal = black_box(next.legal_moves());
        times[2] += start.elapsed().as_nanos();
        drop(legal);
        let start = Instant::now();
        let undo = black_box(next.play_move(&EndTurnMove).expect("EndTurn"));
        times[3] += start.elapsed().as_nanos();
        drop(undo);
        assert_eq!(next.current_player_id(), 2);
    }
    times
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tiles = args.next().context("Tile count")?.parse()?;
    let iterations = args.next().context("Iteration count")?.parse::<usize>()?;
    let game = flat_board(tiles)?;
    let _warmup = run(&game, iterations * 2);
    let [clone, view, legal, end_turn] = run(&game, iterations);
    println!(
        "{}",
        json!({
            "engine": "polyfish",
            "size": game.map_size(),
            "iterations": iterations,
            "legal": game.legal_moves().len(),
            "cloneNs": clone,
            "viewNs": view,
            "legalNs": legal,
            "endTurnNs": end_turn,
        })
    );
    Ok(())
}
