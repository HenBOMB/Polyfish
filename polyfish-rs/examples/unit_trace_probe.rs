//! EXP_ELO_111: `lost_units_probe.rs` says WHICH unit died and WHEN;
//! this says WHY -- every ply that changed one watched unit's coords or
//! health, with the exact target_idx (matches `find_ply_idx`/
//! `attack_pricing_probe3`/`lethality_gate_probe`'s numbering) so a death
//! can be traced straight to the causing command without eyeballing the
//! console summary log (which is a sampled subset, not every played ply).
//!
//! Usage: cargo run --example unit_trace_probe -- <replay.json> <pov> <unit_id>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let pov: i32 = args[2].parse().unwrap();
    let watch_id: u32 = args[3].parse().unwrap();

    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();

    let mut idx = 0usize;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                let legal = game.legal_moves();
                let Some(m) = legal.iter().find(|m| &m.serialize() == cmd) else {
                    continue;
                };
                let this_idx = idx;
                idx += 1;
                let before = game
                    .state
                    .tribes
                    .get(&pov)
                    .and_then(|tr| tr.units.iter().find(|u| u.id == watch_id))
                    .map(|u| (u.coords.idx, u.health));
                game.play_move(m.as_ref());
                let after = game
                    .state
                    .tribes
                    .get(&pov)
                    .and_then(|tr| tr.units.iter().find(|u| u.id == watch_id))
                    .map(|u| (u.coords.idx, u.health));
                if before != after {
                    println!(
                        "idx={this_idx} turn={} player={} cmd={:?} watch_unit before={:?} after={:?}",
                        t.turn, pl.player_id, cmd, before, after
                    );
                }
            }
        }
    }
}
