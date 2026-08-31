//! EXP_ELO_109: does the proposed post-move lethality-exposure gate fire on
//! a given real ply's acting unit? Checks `combat::is_lethally_exposed`
//! PRE-move (at the unit's turn-start coords/health) and POST-move (after
//! `simulate_move`, same frozen `threat_units` snapshot both times) --
//! the gate the fix charges a penalty on is POST true AND PRE false.
//!
//! Usage: cargo run --example lethality_gate_probe -- <replay.json>
//!   <target_idx> <pov> <move_json>
use polyfish::ai::combat;
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let replay_path = &args[1];
    let target_idx: usize = args[2].parse().unwrap();
    let pov: i32 = args[3].parse().unwrap();
    let move_json: serde_json::Value = serde_json::from_str(&args[4]).unwrap();
    let src = move_json["src"].as_i64().unwrap() as i32;

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(pov);
    let threats = combat::threat_units(&view.state, pov);

    let pre_unit = view
        .state
        .tribes
        .get(&pov)
        .unwrap()
        .units
        .iter()
        .find(|u| u.coords.idx == src)
        .unwrap_or_else(|| panic!("no unit at src={src} in pov {pov}'s view"))
        .clone();
    let unit_id = pre_unit.id;
    let pre_w = combat::lethal_threat_weight(&view.state, &pre_unit, &threats);
    let pre_lethal = pre_w > 0.0;
    println!(
        "PRE:  unit_id={unit_id} type={:?} coords={} health={} lethally_exposed={pre_lethal} (w={pre_w:.3})",
        pre_unit.unit_type, pre_unit.coords.idx, pre_unit.health
    );

    let legal = view.legal_moves();
    let m = legal
        .iter()
        .find(|m| &m.serialize() == &move_json)
        .unwrap_or_else(|| panic!("move not legal at this ply: {move_json}"));
    let mut probe = Game { state: view.state.clone() };
    let undo = probe.simulate_move(m.as_ref());

    let post_unit = probe
        .state
        .tribes
        .get(&pov)
        .unwrap()
        .units
        .iter()
        .find(|u| u.id == unit_id)
        .cloned();
    match &post_unit {
        Some(u) => {
            let post_w = combat::lethal_threat_weight(&probe.state, u, &threats);
            let post_lethal = post_w > 0.0;
            println!(
                "POST: unit_id={unit_id} type={:?} coords={} health={} lethally_exposed={post_lethal} (w={post_w:.3})",
                u.unit_type, u.coords.idx, u.health
            );
            let fires = post_lethal && !pre_lethal;
            println!("GATE FIRES: {fires}  (post_lethal={post_lethal}, pre_lethal={pre_lethal})");
        }
        None => println!("POST: unit_id={unit_id} no longer exists (died or was consumed)"),
    }

    if let Some(undo) = undo {
        undo(&mut probe.state);
    }
}
