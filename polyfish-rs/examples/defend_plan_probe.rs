use polyfish::ai::oracle_macro::{commit_macro_goal, StanceCommit, OrderKind};
use polyfish::game::Game;
use polyfish::moves::{AttackMove, Move, StepMove};
use polyfish::replayer::{replay_game, ModReplay};

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
                if idx == target_idx { return game; }
                let legal = game.legal_moves();
                let m = legal.iter().find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    panic!("beyond length");
}

fn dump_plan(label: &str, state: &polyfish::states::GameState, pov: i32, city: i32) {
    let attack_targets: Vec<i32> = vec![];
    let risks = polyfish::ai::combat::city_risks(state, pov);
    let Some(th) = risks.iter().find(|t| t.city == city) else {
        println!("{label}: no CityRisk entry for city {city} (threat cleared?)");
        return;
    };
    println!("{label}: risk={:.3} at_risk={} need_damage={:.3} strike={:.3} attackers={:?}",
        th.risk, th.at_risk, th.need_damage, th.strike, th.attackers);
    let plan = polyfish::ai::combat::defend_plan(state, pov, th, &attack_targets);
    println!("  plan (real):  hold_margin={:.3} shortfall={:.3} assigned={:?}", plan.hold_margin, plan.shortfall, plan.assigned);
    let open = polyfish::ai::combat::defend_plan_open_framing(state, pov, th, &attack_targets, None);
    println!("  plan (open):  hold_margin={:.3} shortfall={:.3} assigned={:?}", open.hold_margin, open.shortfall, open.assigned);
    let garrison = polyfish::functions::get_unit_at(state, city);
    println!("  garrison at {city}: {:?}", garrison.map(|u| (u.owner, u.health, u.unit_type)));
}

fn main() {
    let raw = std::fs::read_to_string("replays/exp101_seed0_watch/game_iter100_game0_seed1787500020.replay.json").unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let true_game = state_at_step(&full, 110);
    dump_plan("PRE-move (real state)", &true_game.state, 1, 49);

    let mv = StepMove::new(61, 49);
    let mut probe = Game { state: true_game.state.clone() };
    let undo = probe.simulate_move(&mv);
    dump_plan("POST Step(61->49)", &probe.state, 1, 49);
    if let Some(u) = undo { u(&mut probe.state); }
}
