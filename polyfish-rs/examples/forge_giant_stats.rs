//! Ad-hoc: replay a batch of self-play games and report, for player 1
//! (the net seat when run with --anchor-seat 2): first turn a Forge is
//! built, first turn its super-unit (Giant for Imperius) count increases,
//! and total super units ever created -- matching self_play's own
//! `tempo.rs::unit_tally`/`giants_made` accounting exactly, since a
//! "Giant" also arrives as an automatic city-level reward, not only via
//! a paid Summon move.
//! Usage: cargo run --example forge_giant_stats -- <replay1.json> <replay2.json> ...
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::settings::units::get_super_unit;
use polyfish::types::{MoveType, StructureType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pov = 1;
    let mut first_forge_turns = Vec::new();
    let mut first_giant_turns = Vec::new();
    let mut total_giants_per_game = Vec::new();
    let mut games_with_forge = 0;
    let mut games_with_giant = 0;

    for path in &args[1..] {
        let raw = std::fs::read_to_string(path).expect("read replay");
        let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
        let mut game = Game::new();
        game.state = full.game_state.clone();
        game.post_load();

        let super_unit = game
            .state
            .tribes
            .get(&pov)
            .map(|t| get_super_unit(t.tribe_type))
            .unwrap();
        let mut prev_giants = 0i32;
        let mut first_forge: Option<i32> = None;
        let mut first_giant: Option<i32> = None;
        let mut total_giants = 0i32;

        for t in &full.turns {
            let mut players: Vec<_> = t.players.iter().collect();
            players.sort_by_key(|p| p.player_id);
            for pl in players {
                for cmd in &pl.commands {
                    let legal = game.legal_moves();
                    let m = legal
                        .iter()
                        .find(|m| &m.serialize() == cmd)
                        .unwrap_or_else(|| panic!("move not legal: {cmd}"));
                    if pl.player_id == pov
                        && m.move_type() == MoveType::Build
                        && m.structure_type().ok() == Some(StructureType::Forge)
                        && first_forge.is_none()
                    {
                        first_forge = Some(game.state.settings.turn);
                    }
                    game.play_move(m.as_ref());
                }
            }
            let giants_now = game
                .state
                .tribes
                .get(&pov)
                .map(|tr| tr.units.iter().filter(|u| u.unit_type == super_unit).count() as i32)
                .unwrap_or(0);
            if giants_now > prev_giants {
                total_giants += giants_now - prev_giants;
                if first_giant.is_none() {
                    first_giant = Some(t.turn);
                }
            }
            prev_giants = giants_now;
        }

        if let Some(ft) = first_forge {
            first_forge_turns.push(ft);
            games_with_forge += 1;
        }
        if let Some(gt) = first_giant {
            first_giant_turns.push(gt);
            games_with_giant += 1;
        }
        total_giants_per_game.push(total_giants);
    }

    let n = args.len() - 1;
    let avg = |v: &[i32]| -> f64 { v.iter().sum::<i32>() as f64 / v.len().max(1) as f64 };
    println!("games: {n}");
    println!(
        "first forge: {games_with_forge}/{n} built one, avg turn (cond) = {:.2}",
        avg(&first_forge_turns)
    );
    println!(
        "first giant: {games_with_giant}/{n} made one, avg turn (cond) = {:.2}",
        avg(&first_giant_turns)
    );
    println!(
        "total giants per game: avg (all games) = {:.3}, avg (cond, >=1) = {:.3}",
        avg(&total_giants_per_game),
        avg(&total_giants_per_game.iter().copied().filter(|&g| g > 0).collect::<Vec<_>>())
    );
}
