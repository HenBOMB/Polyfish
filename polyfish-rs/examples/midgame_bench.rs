use std::{
    hint::black_box,
    io::{self, BufRead, Write},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use polyfish::{
    CityRewardType, CityState, ModeType, MoveType, ResourceState, ResourceType, StructureState,
    StructureType, TechnologyState, TechnologyType, TerrainType, TileState, TribeState, TribeType,
    TurnRotation, UnitState, UnitType,
    ai::mcts::MctsAgent,
    coords::Coords,
    functions::{get_chebyshev_distance, get_city_production},
    game::{Game, set_adversarial_search},
    moves::Move,
};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Scenario {
    version: u32,
    side: i32,
    turn: i32,
    seed: i64,
    stars: i32,
    techs: Vec<String>,
    tiles: Vec<Tile>,
    cities: Vec<City>,
    units: Vec<Unit>,
}

#[derive(Deserialize)]
struct Tile {
    terrain: String,
    resource: Option<String>,
    #[serde(default)]
    road: bool,
}

#[derive(Deserialize)]
struct City {
    seat: i32,
    tile: i32,
    capital: bool,
    production: i32,
}

#[derive(Deserialize)]
struct Unit {
    seat: i32,
    home: i32,
    tile: i32,
    kind: String,
}

type Action = [i32; 3];
const STEP: i32 = 0;
const ATTACK: i32 = 1;
const HARVEST: i32 = 2;
const END: i32 = 3;
const NATIVE_ITERATIONS: usize = 512;

#[derive(Deserialize)]
struct Branch {
    parent: usize,
    action: Action,
}

fn root(scenario: &Scenario, seat: i32, fog: bool) -> Result<Game> {
    let mut game = Game::new();
    let state = &mut game.state;
    let side = scenario.side;
    state.initial_seed = scenario.seed;
    state.settings.seed = scenario.seed;
    state.settings.size = side;
    state.settings.tile_count = side * side;
    state.settings.turn = scenario.turn;
    state.settings.current_player_turn_id = seat;
    state.settings._max_tribe_count = 2;
    state.settings._last_player_turn_id = 3 - seat;
    state.settings._fow = fog;
    state.settings.mode = ModeType::Might;
    state.settings.win_by_capital = true;

    for (id, kind) in [(1, TribeType::XinXi), (2, TribeType::Imperius)] {
        let tech_vanilla = scenario
            .techs
            .iter()
            .map(|name| {
                let tech_type = match name.as_str() {
                    "riding" => TechnologyType::Riding,
                    "roads" => TechnologyType::Roads,
                    "organization" => TechnologyType::Organization,
                    "strategy" => TechnologyType::Strategy,
                    "hunting" => TechnologyType::Hunting,
                    "archery" => TechnologyType::Archery,
                    "climbing" => TechnologyType::Climbing,
                    _ => panic!("Unknown fixture technology {name}"),
                };
                TechnologyState {
                    tech_type,
                    discovered: true,
                    discovered_turn: 1,
                }
            })
            .collect();
        state.tribes.insert(
            id,
            TribeState {
                id,
                tribe_type: kind,
                stars: scenario.stars,
                tech_vanilla,
                known_players: [3 - id].into_iter().collect(),
                ..Default::default()
            },
        );
    }
    for (idx, tile) in scenario.tiles.iter().enumerate() {
        let idx = idx as i32;
        let coords = Coords::from_index(idx, side);
        let mut value = TileState {
            coords,
            terrain_type: match tile.terrain.as_str() {
                "field" => TerrainType::Field,
                "forest" => TerrainType::Forest,
                "mountain" => TerrainType::Mountain,
                "water" => TerrainType::Water,
                terrain => bail!("Unknown fixture terrain {terrain}"),
            },
            has_road: tile.road,
            ..Default::default()
        };
        if !fog || coords.x <= side / 2 + 1 {
            value.explorers.insert(1);
        }
        if !fog || coords.x >= side / 2 - 2 {
            value.explorers.insert(2);
        }
        state.tiles.insert(idx, value);
        if let Some(resource) = &tile.resource {
            let resource_type = match resource.as_str() {
                "fruit" => ResourceType::Fruit,
                "game" => ResourceType::Game,
                resource => bail!("Unknown fixture resource {resource}"),
            };
            state
                .resources
                .insert(idx, Some(ResourceState { resource_type }));
        }
    }
    for city in &scenario.cities {
        let center = Coords::from_index(city.tile, side);
        let tribe = &mut state.tribes[&city.seat];
        if city.capital {
            tribe.starting_tile_coords = center;
            state.tiles[&city.tile].capital_of = city.seat;
        }
        let mut territory = Vec::new();
        for (&idx, tile) in &mut state.tiles {
            if tile.owner == 0 && get_chebyshev_distance(city.tile, idx, side) <= 1 {
                tile.owner = city.seat;
                tile.ruling_city_coords = Some(center);
                territory.push(idx);
            }
        }
        tribe.cities.push(CityState {
            owner: city.seat,
            idx: city.tile,
            level: 3,
            rewards: vec![CityRewardType::Explorer, CityRewardType::Resources],
            production: city.production - i32::from(city.capital),
            _territory: territory,
            ..Default::default()
        });
        state.structures.insert(
            city.tile,
            Some(StructureState {
                structure_type: StructureType::Village,
                ..Default::default()
            }),
        );
    }
    for unit in &scenario.units {
        let (unit_type, health) = match unit.kind.as_str() {
            "warrior" => (UnitType::Warrior, 10.0),
            "rider" => (UnitType::Rider, 10.0),
            "archer" => (UnitType::Archer, 10.0),
            "defender" => (UnitType::Defender, 15.0),
            kind => bail!("Unknown fixture unit {kind}"),
        };
        let coords = Coords::from_index(unit.tile, side);
        state.tiles[&unit.tile]._unit_owner_id = Some(unit.seat);
        state.tribes[&unit.seat].units.push(UnitState {
            owner: unit.seat,
            unit_type,
            health,
            coords,
            prev_coords: coords,
            home_coords: Some(Coords::from_index(unit.home, side)),
            ..Default::default()
        });
    }
    polyfish::functions::sync_scores(state);
    Ok(game)
}

fn descriptor(action: &dyn Move) -> Option<Action> {
    match action.move_type() {
        MoveType::Step => Some([
            STEP,
            action.source_idx().ok()? as i32,
            action.target_idx().ok()? as i32,
        ]),
        MoveType::Attack => Some([
            ATTACK,
            action.source_idx().ok()? as i32,
            action.target_idx().ok()? as i32,
        ]),
        MoveType::Harvest => Some([HARVEST, -1, action.target_idx().ok()? as i32]),
        MoveType::EndTurn => Some([END, -1, -1]),
        _ => None,
    }
}

fn projection(game: &Game) -> Value {
    let state = &game.state;
    let mut cities = Vec::new();
    let mut units = Vec::new();
    for (seat, tribe) in &state.tribes {
        for city in &tribe.cities {
            cities.push([
                city.idx,
                *seat,
                city.progress,
                get_city_production(state, city),
            ]);
        }
        for unit in &tribe.units {
            let kind = match unit.unit_type {
                UnitType::Warrior => 0,
                UnitType::Rider => 1,
                UnitType::Archer => 2,
                UnitType::Defender => 3,
                _ => -1,
            };
            units.push([unit.coords.idx, *seat, kind, unit.health as i32]);
        }
    }
    cities.sort_unstable();
    units.sort_unstable();
    let land: Vec<_> = state
        .tiles
        .iter()
        .map(|(&idx, tile)| {
            let terrain = match tile.terrain_type {
                TerrainType::Field => 0,
                TerrainType::Forest => 1,
                TerrainType::Mountain => 2,
                TerrainType::Water => 3,
                _ => -1,
            };
            let resource = match state
                .resources
                .get(&idx)
                .and_then(Option::as_ref)
                .map(|r| r.resource_type)
            {
                Some(ResourceType::Fruit) => 1,
                Some(ResourceType::Game) => 2,
                _ => 0,
            };
            [terrain, tile.owner, resource, i32::from(tile.has_road)]
        })
        .collect();
    json!({
        "seat": game.current_player_id(), "turn": game.turn(),
        "stars": [state.tribes[&1].stars, state.tribes[&2].stars],
        "cities": cities, "units": units, "land": land,
    })
}

fn expansion(parent: &Game, wanted: Action, times: &mut [u128; 4], timed: bool) -> Result<Game> {
    let start = timed.then(Instant::now);
    let mut child = parent.clone();
    if let Some(start) = start {
        times[0] += start.elapsed().as_nanos();
    }
    let start = timed.then(Instant::now);
    let decision = child.decision();
    if let Some(start) = start {
        times[1] += start.elapsed().as_nanos();
    }
    let start = timed.then(Instant::now);
    let index = decision
        .moves()
        .iter()
        .position(|action| descriptor(action.as_ref()) == Some(wanted))
        .with_context(|| {
            format!(
                "Missing action {wanted:?} on turn {}",
                decision.game().turn()
            )
        })?;
    if let Some(start) = start {
        times[2] += start.elapsed().as_nanos();
    }
    let start = timed.then(Instant::now);
    drop(
        decision
            .simulate_index(index, TurnRotation::NextPlayer)
            .with_context(|| format!("Could not apply {wanted:?}"))?,
    );
    if let Some(start) = start {
        times[3] += start.elapsed().as_nanos();
    }
    Ok(child)
}

fn oracle(root: Game) -> Result<()> {
    let mut states = vec![root];
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let command: Value = serde_json::from_str(&line?)?;
        let node = command["node"].as_u64().context("Missing node")? as usize;
        let response = match command["op"].as_str() {
            Some("inspect") => {
                let decision = states[node].decision();
                let legal = decision.moves();
                let actions: Vec<_> = legal
                    .iter()
                    .filter_map(|action| descriptor(action.as_ref()))
                    .collect();
                json!({"projection": projection(decision.game()), "legal": legal.len(), "actions": actions})
            }
            Some("expand") => {
                let wanted: Action = serde_json::from_value(command["action"].clone())?;
                let result = expansion(&states[node], wanted, &mut [0; 4], false)?;
                let projected = projection(&result);
                let slot = command["slot"].as_u64().context("Missing slot")? as usize;
                if slot == states.len() {
                    states.push(result);
                } else {
                    states[slot] = result;
                }
                projected
            }
            _ => bail!("Unknown oracle command"),
        };
        serde_json::to_writer(&mut out, &response)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}

fn search_batch(
    root: &Game,
    schedule: &[Branch],
    times: &mut [u128; 4],
    timed: bool,
) -> Result<()> {
    let mut held = Vec::with_capacity(schedule.len() + 1);
    held.push(root.clone());
    for branch in schedule {
        let child = expansion(&held[branch.parent], branch.action, times, timed)?;
        held.push(child);
    }
    black_box(&held);
    Ok(())
}

fn native(roots: &[Game; 2], iterations: usize) -> Result<()> {
    let agent = MctsAgent::new(NATIVE_ITERATIONS);
    for iteration in 0..iterations {
        let root = &roots[iteration % roots.len()];
        let mut view = root.clone_for_mcts(root.current_player_id());
        let move_choice = agent
            .select_move(&mut view)
            .context("MCTS did not select an action")?;
        black_box(move_choice);
    }
    Ok(())
}

fn main() -> Result<()> {
    set_adversarial_search(true);
    let args: Vec<_> = std::env::args().skip(1).collect();
    let scene: Scenario = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    anyhow::ensure!(scene.version == 1 && scene.tiles.len() == (scene.side * scene.side) as usize);
    let mode = args[1].as_str();
    if mode == "oracle" {
        return oracle(root(&scene, 1, false)?);
    }
    let count: usize = args[2].parse()?;
    let warmup: usize = args[3].parse()?;
    let mut row = match mode {
        "matched" | "components" | "retain" => {
            let schedule: Vec<Branch> = serde_json::from_slice(&std::fs::read(&args[4])?)?;
            let game = root(&scene, 1, false)?;
            if mode == "retain" {
                search_batch(&game, &schedule[..count], &mut [0; 4], false)?;
                json!({"retained": count})
            } else {
                let schedule = &schedule[..512];
                for _ in 0..warmup {
                    search_batch(&game, schedule, &mut [0; 4], false)?;
                }
                let mut times = [0; 4];
                let start = Instant::now();
                for _ in 0..count {
                    search_batch(&game, schedule, &mut times, mode == "components")?;
                }
                json!({"nanos": start.elapsed().as_nanos(), "expansions": count * schedule.len(),
                       "copyNs": times[0], "legalNs": times[1], "lookupNs": times[2], "applyNs": times[3]})
            }
        }
        "native" => {
            let roots = [root(&scene, 1, true)?, root(&scene, 2, true)?];
            native(&roots, warmup)?;
            let start = Instant::now();
            native(&roots, count)?;
            json!({"nanos": start.elapsed().as_nanos(), "decisions": count,
                   "iterationsPerDecision": NATIVE_ITERATIONS})
        }
        _ => bail!("Unknown midgame benchmark mode {mode}"),
    };
    row["engine"] = json!("polyfish");
    row["size"] = json!(scene.side);
    println!("{row}");
    Ok(())
}
