//! Ad-hoc: print site_value/city_build_on internals for named hub-site
//! candidates on a real --state dump, to see the delivered spt/giants/
//! stars/partners eco_plan actually computes per site (not just the
//! display-only partner-count list in `explain()`).
//! Usage: cargo run --example hub_site_probe2 -- <state.json> <city_idx> <sites...>

use polyfish::game::Game;
use polyfish::rules::eco_plan::*;
use polyfish::states::GameState;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let state_path = &args[1];
    let city_idx: i32 = args[2].parse().unwrap();
    let sites: Vec<i32> = args[3..].iter().map(|s| s.parse().unwrap()).collect();

    let raw = std::fs::read_to_string(state_path).expect("read state");
    let loaded: GameState = serde_json::from_str(&raw).expect("parse state");
    let mut game = Game::new();
    game.state = loaded;
    game.post_load();

    let pov = game.state.settings.current_player_turn_id;
    let cities: Vec<i32> = game
        .state
        .tribes
        .get(&pov)
        .map(|t| t.cities.iter().map(|c| c.idx).collect())
        .unwrap_or_default();
    println!("pov {pov} cities {cities:?}");

    for sc in SCENARIOS.iter().filter(|s| s.lane == Lane::Mine) {
        // Use the same territory allocation the real --explain path uses.
        let terr = allocate_value(&game.state, &cities, &uniform(*sc, cities.len()), 0);
        let ci = cities.iter().position(|&c| c == city_idx).expect("city not held");
        let territory = &terr[ci];
        println!("\n=== {} | territory {:?} ===", sc.name, territory);

        let plot = Plot::new(&game.state, territory, *sc);
        println!("  plot.hub_sites: {:?}", plot.hub_sites);
        println!("  plot.partner_tiles (own-territory only): {:?}", plot.partner_tiles);

        for &site in &sites {
            if !plot.hub_sites.contains(&site) {
                println!("  site {site}: NOT a legal hub_site for this scenario/territory");
                continue;
            }
            let b = city_build_on(&game.state, &plot, *sc, 0, Some(site), None, None, &[]);
            let (spt, giants, stars, pop) =
                site_value(&game.state, city_idx, territory, *sc, 0, Some(site));
            let reachable = site_reachable(&game.state, city_idx, territory, *sc, 0, site);
            println!(
                "  site {site:4}: partners(city_build_on)={:2} pop={:3} stars={:3} | site_value: spt={spt} giants={giants} stars={stars} pop={pop} | reachable={reachable}",
                b.partners, b.pop, b.stars
            );
        }

        let candidates = hub_candidates(&game.state, territory, *sc, HUB_TOP_K);
        println!("  hub_candidates (own-territory partner-count order): {:?}", candidates);
        let chosen = build_out(&game.state, city_idx, territory, *sc, 0, Goal::Balanced);
        println!("  build_out chosen hub: {:?} partners={} pop={} stars={}", chosen.hub_site, chosen.partners, chosen.pop, chosen.stars);
    }
    let _ = HashSet::<i32>::new();
}
