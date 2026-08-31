//! EXP_ELO_112 verification: reconstructs the exact `oracle_macro::OrderKind
//! ::Attack` gate inputs against a target city -- our clustered value within
//! Chebyshev 3, their defenders within Chebyshev 2, and whether the 1.5x
//! margin clears -- so a "we should be attacking" claim can be checked
//! against the real gate instead of eyeballed total army size.
//! Usage: cargo run --example army_pressure_probe -- <replay.json> <target_idx> <pov> <enemy_pov> <target_city>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let target_idx: usize = args[2].parse().unwrap();
    let pov: i32 = args[3].parse().unwrap();
    let enemy_pov: i32 = args[4].parse().unwrap();
    let target_city: i32 = args[5].parse().unwrap();
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
    let cheb =
        |a: i32, b: i32| polyfish::functions::get_chebyshev_distance(a, b, game.state.settings.size);
    let explored = game
        .state
        .tiles
        .get(&target_city)
        .map_or(false, |t| t.explorers.contains(&pov));
    println!("tile{target_city} explored_by_p{pov}={explored}");
    let unit_cost = |u: &polyfish::states::UnitState| polyfish::rules::combat::unit_worth(u);
    let tribe = game.state.tribes.get(&pov).unwrap();
    let own_units: Vec<(i32, i32)> = tribe.units.iter().map(|u| (u.coords.idx, unit_cost(u))).collect();
    let our_army: i32 = own_units.iter().map(|(_, c)| c).sum();
    let local: Vec<i32> = own_units
        .iter()
        .filter(|(u, _)| cheb(*u, target_city) <= 3)
        .map(|(_, c)| *c)
        .collect();
    println!(
        "our_army_total={our_army} units_within3={} value_within3={}",
        local.len(),
        local.iter().sum::<i32>()
    );
    if let Some(enemy) = game.state.tribes.get(&enemy_pov) {
        let defenders: i32 = enemy
            .units
            .iter()
            .filter(|u| cheb(u.coords.idx, target_city) <= 2)
            .map(unit_cost)
            .sum();
        let margin_needed = 1.5 * defenders as f32;
        let gate_fires = local.len() >= 2 && 2 * local.iter().sum::<i32>() > 3 * defenders;
        println!(
            "enemy_defenders_within2={defenders} margin_needed(1.5x)={margin_needed:.1} attack_gate_fires={gate_fires}"
        );
        let enemy_army: i32 = enemy.units.iter().map(unit_cost).sum();
        println!("enemy_total_army_value={enemy_army}");
    }
}
