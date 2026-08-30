//! KPI snapshot for one game against Verdi's four success metrics:
//! turn game ends, units lost/killed for the net seat, giants by turn 12.
//! Usage: cargo run --example game_kpis -- <replay.json> [pov=1]
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::settings::units::get_super_unit;
use polyfish::types::MoveType;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let pov: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let opp = if pov == 1 { 2 } else { 1 };

    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();

    let super_unit = game.state.tribes.get(&pov).map(|t| get_super_unit(t.tribe_type)).unwrap();
    let mut prev_giants = 0i32;
    let mut giants_by_turn12 = 0i32;
    let mut units_lost = 0i32;
    let mut units_killed = 0i32; // opponent units lost, credited to us
    let mut prev_pov_units = game.state.tribes.get(&pov).map(|t| t.units.len()).unwrap_or(0) as i32;
    let mut prev_opp_units = game.state.tribes.get(&opp).map(|t| t.units.len()).unwrap_or(0) as i32;
    let mut last_turn = 0;

    for t in &full.turns {
        last_turn = t.turn;
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                let legal = game.legal_moves();
                let m = legal.iter().find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("not legal: {cmd}"));
                let _ = m.move_type();
                game.play_move(m.as_ref());
            }
        }
        let pov_units = game.state.tribes.get(&pov).map(|t| t.units.len()).unwrap_or(0) as i32;
        let opp_units = game.state.tribes.get(&opp).map(|t| t.units.len()).unwrap_or(0) as i32;
        // Coarse (unit-count based, not perfectly move-attributed): a drop
        // in our count this turn is a loss; a drop in theirs is a kill.
        if pov_units < prev_pov_units {
            units_lost += prev_pov_units - pov_units;
        }
        if opp_units < prev_opp_units {
            units_killed += prev_opp_units - opp_units;
        }
        prev_pov_units = pov_units;
        prev_opp_units = opp_units;

        let giants_now = game.state.tribes.get(&pov)
            .map(|tr| tr.units.iter().filter(|u| u.unit_type == super_unit).count() as i32)
            .unwrap_or(0);
        if giants_now > prev_giants && t.turn <= 12 {
            giants_by_turn12 += giants_now - prev_giants;
        }
        prev_giants = giants_now;
    }

    let is_game_over = polyfish::functions::is_game_over(&game.state);
    println!("last recorded turn: {last_turn}");
    println!("game_over at end of replay: {is_game_over}");
    println!("units lost (pov {pov}): {units_lost}");
    println!("units killed (opponent {opp}): {units_killed}");
    println!("giants by turn 12: {giants_by_turn12}");
}
