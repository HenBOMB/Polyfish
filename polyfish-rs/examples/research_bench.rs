use std::{hint::black_box, time::Instant};

use anyhow::{Context, Result};
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::moves::{EndTurnMove, StepMove};
use polyfish::research::{self, MatchConfig, Opponent};
use polyfish::states::StructureState;
use polyfish::types::{MoveType, StructureType, TerrainType, TribeType};
use serde_json::json;

const TRACE_ROUNDS: i32 = 10;

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
        game.state.structures.insert(
            index,
            Some(StructureState {
                structure_type: StructureType::Village,
                ..StructureState::default()
            }),
        );
    }
    game.post_load();
    Ok(game)
}

fn run(game: &Game, iterations: usize) -> [u128; 5] {
    let mut times = [0; 5];
    let step = StepMove::new(game.map_size() + 1, 2 * (game.map_size() + 1));
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

        let mut moving = game.clone();
        let start = Instant::now();
        let undo = black_box(moving.play_move(&step).expect("Step"));
        times[4] += start.elapsed().as_nanos();
        drop(undo);
        assert_eq!(
            moving.state.tribes[&1].units[0].coords.idx,
            step.target_index
        );
    }
    times
}

fn run_traces(game: &Game, iterations: usize) {
    let stride = game.map_size() + 1;
    let capitals = [stride, (game.map_size() - 2) * stride];
    let nearby = [2 * stride, (game.map_size() - 3) * stride];
    for _ in 0..iterations {
        let mut state = game.clone();
        for round in 0..TRACE_ROUNDS {
            for seat in 0..2 {
                assert_eq!(state.current_player_id(), seat as i32 + 1);
                let target = if round % 2 == 0 {
                    nearby[seat]
                } else {
                    capitals[seat]
                };
                let step = state
                    .legal_moves()
                    .into_iter()
                    .find(|action| {
                        action.move_type() == MoveType::Step
                            && action.target_idx().ok() == Some(target as usize)
                    })
                    .expect("Scripted step");
                let _ = state.play_move(step.as_ref()).expect("Apply step");
                assert_eq!(
                    state.state.tribes[&(seat as i32 + 1)].units[0].coords.idx,
                    target
                );
                let end_turn = state
                    .legal_moves()
                    .into_iter()
                    .find(|action| action.move_type() == MoveType::EndTurn)
                    .expect("Scripted end turn");
                let _ = state.play_move(end_turn.as_ref()).expect("Apply end turn");
            }
        }
        assert_eq!(state.current_player_id(), 1);
        assert_eq!(state.turn(), game.turn() + TRACE_ROUNDS);
        black_box(state);
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let tiles = args.next().context("Tile count")?.parse()?;
    let iterations = args.next().context("Iteration count")?.parse::<usize>()?;
    let mode = args.next().unwrap_or_else(|| "ops".into());
    let warmup = args
        .next()
        .map(|count| count.parse())
        .transpose()?
        .unwrap_or(iterations * 2);
    let game = flat_board(tiles)?;
    let data = match mode.as_str() {
        "ops" => {
            run(&game, warmup);
            let [clone, view, legal, end_turn, step] = run(&game, iterations);
            json!({
                "iterations": iterations, "legal": game.legal_moves().len(),
                "cloneNs": clone, "viewNs": view, "legalNs": legal, "endTurnNs": end_turn, "stepNs": step,
            })
        }
        "trace" => {
            run_traces(&game, warmup);
            let start = Instant::now();
            run_traces(&game, iterations);
            json!({
                "iterations": iterations, "actionsPerTrace": 4 * TRACE_ROUNDS,
                "traceNs": start.elapsed().as_nanos(),
            })
        }
        "retain" => {
            let copies: Vec<_> = (0..iterations).map(|_| game.clone()).collect();
            black_box(&copies);
            json!({"retainedCopies": copies.len()})
        }
        _ => anyhow::bail!("Unknown benchmark mode {mode}"),
    };
    let mut output = data;
    output["engine"] = json!("polyfish");
    output["size"] = json!(game.map_size());
    println!("{output}");
    Ok(())
}
