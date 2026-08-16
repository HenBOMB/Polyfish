//! EXP_ELO_032: goal-conditioned deterministic whole-turn executor. Given a
//! fixed `MacroGoal`, plays out one player's turn ply-by-ply on a rollout
//! clone — the micro half of the macro-search bootstrap. Reuses the
//! oracle_macro root gates and prices plies as
//! `score_move + λ·Δgoal_potential` (the edge_snapshot pattern).

use crate::ai::oracle_macro::{
    ArchetypeState, GoalAux, MacroGoal, Stance, tech_discipline_active, passes_ability_gate,
    passes_capture_first, passes_stance_tech_mask, passes_tech_purchase_limits, compute_goal_aux,
    observe_archetype,
};
use crate::ai::{reward, scoring};
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::{GameState, PlayerId};
use crate::types::MoveType;

/// Whole-game purchase counters the tech-cap gates read (mirrors the
/// goal_script counting in arena.rs / self_play.rs).
#[derive(Clone, Copy, Debug, Default)]
pub struct TurnCounters {
    pub techs_bought: u32,
    pub tier3_bought: u32,
}

impl TurnCounters {
    /// Count `m` if it is a Research purchase (tier-3 tracked separately).
    pub fn count(&mut self, m: &dyn Move) {
        if m.move_type() != MoveType::Research {
            return;
        }
        self.techs_bought += 1;
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::get_technology_setting(tech).tier == Some(3) {
                self.tier3_bought += 1;
            }
        }
    }
}

/// The four oracle_macro root gates, EndTurn always exempt (mirrors
/// gumbel_mcts::gate_retain, which stays private to keep its attribution
/// counters off this path).
pub fn gate_ok(
    state: &GameState,
    m: &dyn Move,
    star_gate: bool,
    stance: Option<Stance>,
    aux: Option<&GoalAux>,
) -> bool {
    if m.move_type() == MoveType::EndTurn {
        return true;
    }
    if star_gate && !passes_stance_tech_mask(state, m, stance, aux) {
        return false;
    }
    if let Some(a) = aux {
        if !passes_tech_purchase_limits(m, a)
            || !passes_ability_gate(state, m)
            || !passes_capture_first(state, m)
        {
            return false;
        }
    }
    true
}

/// Rank the current player's plies under a fixed goal, best first. EndTurn is
/// suppressed while any other move survives the gates (the search backends'
/// root convention); a fully gated-out ply degrades to a lone EndTurn.
/// Δφ probes one move at a time via simulate_move/undo — single-move undo is
/// safe; composed undos across a turn are not (see cross_end_turn).
pub fn rank_plies(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    aux: &GoalAux,
    star_gate: bool,
    lambda: f32,
) -> Vec<(f32, Box<dyn Move>)> {
    let mut moves = game.legal_moves();
    moves.retain(|m| gate_ok(&game.state, m.as_ref(), star_gate, Some(goal.stance), Some(aux)));
    let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        moves.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if moves.is_empty() {
        return vec![(0.0, Box::new(EndTurnMove) as Box<dyn Move>)];
    }

    let phi_pre = if lambda != 0.0 {
        reward::goal_potential(&game.state, player, goal, Some(aux))
    } else {
        0.0
    };
    let mut scored: Vec<(f32, Box<dyn Move>)> = moves
        .into_iter()
        .map(|m| {
            let mut s = scoring::score_move(game, m.as_ref());
            if lambda != 0.0 && m.move_type() != MoveType::EndTurn {
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let phi_post = reward::goal_potential(&game.state, player, goal, Some(aux));
                    undo(&mut game.state);
                    s += lambda * (phi_post - phi_pre);
                }
            }
            (s, m)
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored
}

/// Ply cap per executed turn, mirroring cross_end_turn's MAX_GHOST_MOVES.
pub const MAX_EXEC_PLIES: usize = 64;

/// Execute one full turn for `player` on a ROLLOUT CLONE under a fixed goal,
/// ending with `simulate_single_end_turn` (never the blind opponent-skipping
/// `simulate_move(EndTurn)`). Returns false on an anomaly (rejected move) —
/// the caller scores the rollout where it stopped.
pub fn execute_turn(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    arch: &mut ArchetypeState,
    counters: &mut TurnCounters,
    lambda: f32,
) -> bool {
    execute_turn_recorded(game, player, goal, arch, counters, lambda, None)
}

/// One executed ply, for the Tier-2/Tier-3 boundary probe (EXP_ELO_048).
/// `flip_no_phi` / `flip_no_goal` answer "would this ply have been chosen
/// anyway", with the directive's ranking term removed and with the directive
/// removed entirely — i.e. how much of the executor's choice the directive
/// actually owns.
#[derive(Clone, Debug)]
pub struct PlyRec {
    pub mv: String,
    pub kind: String,
    pub flip_no_phi: bool,
    pub flip_no_goal: bool,
}

/// `execute_turn` with an optional per-ply recorder. One implementation, so a
/// probe can never drift from the executor it is measuring; the recording
/// arms cost two extra `rank_plies` per ply and are OFF unless `rec` is set.
pub fn execute_turn_recorded(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    arch: &mut ArchetypeState,
    counters: &mut TurnCounters,
    lambda: f32,
    mut rec: Option<&mut Vec<PlyRec>>,
) -> bool {
    for _ in 0..MAX_EXEC_PLIES {
        if game.state.settings._game_over
            || game.state.settings.current_player_turn_id != player
        {
            return true;
        }
        observe_archetype(&game.state, player, arch);
        let aux = compute_goal_aux(
            &game.state,
            player,
            goal,
            counters.techs_bought,
            counters.tier3_bought,
            Some(arch),
        );
        let gate = tech_discipline_active(&game.state, player, goal);
        let mut ranked = rank_plies(game, player, goal, &aux, gate, lambda);
        if ranked.is_empty() {
            break;
        }
        let (_, best) = ranked.swap_remove(0);
        if best.move_type() == MoveType::EndTurn {
            break;
        }
        if let Some(r) = rec.as_deref_mut() {
            let key = best.serialize().to_string();
            // Same gate, no directive PULL: isolates the lambda*dphi term.
            let no_phi = rank_plies(game, player, goal, &aux, gate, 0.0);
            // No directive at all: gate open, default goal — the whole Tier-2
            // channel removed, both filter and pull.
            let bare = MacroGoal::default();
            let bare_aux = compute_goal_aux(
                &game.state,
                player,
                &bare,
                counters.techs_bought,
                counters.tier3_bought,
                Some(arch),
            );
            let no_goal = rank_plies(game, player, &bare, &bare_aux, false, 0.0);
            let top = |v: &Vec<(f32, Box<dyn Move>)>| {
                v.first().map(|(_, m)| m.serialize().to_string()).unwrap_or_default()
            };
            r.push(PlyRec {
                kind: format!("{:?}", best.move_type()),
                flip_no_phi: top(&no_phi) != key,
                flip_no_goal: top(&no_goal) != key,
                mv: key,
            });
        }
        if game.simulate_move(best.as_ref()).is_none() {
            return false;
        }
        counters.count(best.as_ref());
    }
    let _ = game.simulate_single_end_turn();
    true
}

/// Ghost-play every intervening opponent with deterministic Greedy until
/// control returns to `pov` or the game ends (cross_end_turn's loop, minus
/// snapshot/undo — the caller's clone is throwaway). A runaway opponent turn
/// is force-ended so the rollout keeps advancing; returns false only on an
/// anomaly (no move / rejected move), where the caller scores the state as-is.
pub fn ghost_until(game: &mut Game, pov: PlayerId) -> bool {
    while game.state.settings.current_player_turn_id != pov && !game.state.settings._game_over {
        let mut n = 0usize;
        loop {
            if game.state.settings._game_over {
                return true;
            }
            if n >= MAX_EXEC_PLIES {
                let _ = game.simulate_single_end_turn();
                break;
            }
            match crate::ai::heuristic_mcts::GreedyHeuristicAgent.select_move(game) {
                None => return false,
                Some(mv) if mv.move_type() == MoveType::EndTurn => {
                    let _ = game.simulate_single_end_turn();
                    break;
                }
                Some(mv) => {
                    if game.simulate_move(mv.as_ref()).is_none() {
                        return false;
                    }
                }
            }
            n += 1;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::oracle_macro::{compute_macro_goal, commit_macro_goal, StanceCommit};

    fn generated_game(seed: i64) -> Game {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        game
    }

    #[test]
    fn executor_terminates_and_hands_over() {
        for seed in 0..8i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let goal = compute_macro_goal(&sim.state, pov, 0, None);
            let mut arch = ArchetypeState::default();
            let mut counters = TurnCounters::default();
            let ok = execute_turn(&mut sim, pov, &goal, &mut arch, &mut counters, 1.0);
            assert!(ok, "seed {seed}: executor bailed on its own turn");
            assert!(
                sim.state.settings.current_player_turn_id != pov
                    || sim.state.settings._game_over,
                "seed {seed}: turn never handed over"
            );
        }
    }

    #[test]
    fn executor_is_deterministic() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut history = Vec::new();
            for _ in 0..2 {
                let mut sim = game.clone_for_mcts(pov);
                let goal = compute_macro_goal(&sim.state, pov, 0, None);
                let mut arch = ArchetypeState::default();
                let mut counters = TurnCounters::default();
                execute_turn(&mut sim, pov, &goal, &mut arch, &mut counters, 1.0);
                history.push(sim.state._history.clone());
            }
            assert_eq!(history[0], history[1], "seed {seed}: two runs diverged");
        }
    }

    #[test]
    fn gates_never_strand_the_turn() {
        for seed in 0..4i64 {
            let mut game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut commit = StanceCommit::default();
            let goal = commit_macro_goal(&game.state, pov, &mut commit, 0);
            // Saturated caps: every Research is gated, the turn must still end.
            let aux = compute_goal_aux(
                &game.state,
                pov,
                &goal,
                crate::ai::oracle_macro::TECH_CAP_PER_GAME,
                crate::ai::oracle_macro::TIER3_CAP_PER_GAME,
                None,
            );
            let ranked = rank_plies(&mut game, pov, &goal, &aux, true, 1.0);
            assert!(!ranked.is_empty(), "seed {seed}: rank_plies returned empty");
            assert!(
                ranked
                    .iter()
                    .all(|(_, m)| m.move_type() != MoveType::Research),
                "seed {seed}: gated Research survived"
            );
        }
    }

    #[test]
    fn ghost_until_returns_control() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let goal = compute_macro_goal(&sim.state, pov, 0, None);
            let mut arch = ArchetypeState::default();
            let mut counters = TurnCounters::default();
            assert!(execute_turn(&mut sim, pov, &goal, &mut arch, &mut counters, 1.0));
            if sim.state.settings._game_over {
                continue;
            }
            assert!(ghost_until(&mut sim, pov), "seed {seed}: ghost bailed");
            assert!(
                sim.state.settings.current_player_turn_id == pov
                    || sim.state.settings._game_over,
                "seed {seed}: control never returned"
            );
        }
    }
}
