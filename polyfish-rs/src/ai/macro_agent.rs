//! EXP_ELO_032: the two macro-bootstrap agents.
//! `MacroScriptAgent` (Stage 0) = scripted directive + deterministic executor.
//! `MacroLookaheadAgent` (Stage 1) = enumerate K candidate directives at each
//! own-turn start, roll each out H turns on FOW-honest clones (executor for
//! own turns, ghost Greedy for the opponent), score leaves, commit the winner
//! for the whole turn. Inference-only — nothing here touches training.

use crate::ai::eval_server::Evaluator;
use crate::ai::macro_exec::{self, TurnCounters};
use crate::ai::oracle_macro::{
    ArchetypeState, MacroGoal, OrderKind, Stance, StanceCommit, goal_star_gate,
    retakeable_village, save_batch_plan, scripted_goal, scripted_goal_aux, still_capturable,
    update_archetype, update_goal,
};
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::{GameState, PlayerId};

/// Leaf scorer for the Stage-1 rollouts.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MacroLeaf {
    Heuristic,
    Net,
}

#[derive(Clone, Copy, Debug)]
pub struct MacroParams {
    /// Max candidate directives per turn (the scripted base is always kept).
    pub k: usize,
    /// Own turns simulated per rollout, including the candidate turn.
    pub horizon: u32,
    pub leaf: MacroLeaf,
    /// λ on Δgoal_potential in ply ranking (production GOAL_W_TREE=1; φ is
    /// score-equivalent, so 1.0 is calibrated, not arbitrary).
    pub lambda: f32,
    /// EXP_ELO_033: simulations per turn-level tree search (macro-mcts only).
    pub sims: usize,
}

impl Default for MacroParams {
    fn default() -> Self {
        Self { k: 4, horizon: 2, leaf: MacroLeaf::Heuristic, lambda: 1.0, sims: 32 }
    }
}

/// Plans are made on the fogged view, so a chosen move can be illegal on the
/// true state (arena ignores play_move failure and re-asks on unchanged state
/// — a livelock). Walk the ranked list intersecting against the true game's
/// legal set keyed by `serialize()` (the move-identity key); EndTurn is the
/// guaranteed fallback.
pub(crate) fn first_true_legal(
    game: &Game,
    ranked: Vec<(f32, Box<dyn Move>)>,
) -> Box<dyn Move> {
    let legal: std::collections::HashSet<String> = game
        .legal_moves()
        .iter()
        .map(|m| m.serialize().to_string())
        .collect();
    ranked
        .into_iter()
        .map(|(_, m)| m)
        .find(|m| legal.contains(&m.serialize().to_string()))
        .unwrap_or_else(|| Box::new(EndTurnMove))
}

/// Stage 0: Greedy conditioned on the scripted directive. Per-game state —
/// arena constructs a fresh agent per match, so counters reset per game.
pub struct MacroScriptAgent {
    lambda: f32,
    stance_commit: StanceCommit,
    archetype: ArchetypeState,
    counters: TurnCounters,
}

impl MacroScriptAgent {
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda,
            stance_commit: StanceCommit::default(),
            archetype: ArchetypeState::default(),
            counters: TurnCounters::default(),
        }
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = update_goal(&view.state, pov, &mut self.stance_commit, self.counters.tier3_bought);
        let ranked = rank_view(&mut view, pov, &goal, &mut self.archetype, &mut self.counters, self.lambda);
        let m = first_true_legal(game, ranked);
        self.counters.count(m.as_ref());
        Some(m)
    }
}

/// One ply of goal-conditioned ranking on the fogged view (shared by the
/// macro agents; the goal differs — fresh scripted vs the turn's committed
/// winner).
pub(crate) fn rank_view(
    view: &mut Game,
    pov: PlayerId,
    goal: &MacroGoal,
    archetype: &mut ArchetypeState,
    counters: &mut TurnCounters,
    lambda: f32,
) -> Vec<(f32, Box<dyn Move>)> {
    update_archetype(&view.state, pov, goal, archetype);
    let aux = scripted_goal_aux(
        &view.state,
        pov,
        goal,
        counters.techs_bought,
        counters.tier3_bought,
        Some(archetype),
    );
    let gate = goal_star_gate(&view.state, pov, goal);
    macro_exec::rank_plies(view, pov, goal, &aux, gate, lambda)
}

/// Candidate directive set: the committed script base first, then stance
/// overrides and order variants. Orders stay sorted (feature-byte hashing
/// invariant) and duplicates collapse, so `k` buys distinct strategies.
pub fn enumerate_candidates(
    state: &GameState,
    pov: PlayerId,
    base: MacroGoal,
    counters: TurnCounters,
    k: usize,
) -> Vec<MacroGoal> {
    let mut out: Vec<MacroGoal> = vec![base.clone()];
    let push = |mut g: MacroGoal, out: &mut Vec<MacroGoal>| {
        g.orders.sort();
        if !out.contains(&g) {
            out.push(g);
        }
    };

    push(
        MacroGoal { orders: base.orders.clone(), stance: Stance::Grow, save_target: None },
        &mut out,
    );
    push(
        MacroGoal { orders: base.orders.clone(), stance: Stance::Arm, save_target: None },
        &mut out,
    );
    if let Some(lane) = save_batch_plan(state, pov, counters.tier3_bought) {
        push(
            MacroGoal {
                orders: base.orders.clone(),
                stance: Stance::Save,
                save_target: Some(lane),
            },
            &mut out,
        );
    }
    // Real capturable/retakeable targets only — drops generator-guessed sites.
    let real: Vec<(OrderKind, i32)> = base
        .orders
        .iter()
        .filter(|(kind, idx)| {
            *kind != OrderKind::Expand
                || still_capturable(state, *idx, pov)
                || retakeable_village(state, *idx, pov)
        })
        .cloned()
        .collect();
    if real != base.orders {
        push(
            MacroGoal { orders: real, stance: base.stance, save_target: base.save_target.clone() },
            &mut out,
        );
    }
    if let Some(cap) = explored_enemy_capital(state, pov) {
        if !base.orders.contains(&(OrderKind::Attack, cap)) {
            let mut orders = base.orders.clone();
            orders.push((OrderKind::Attack, cap));
            push(
                MacroGoal { orders, stance: base.stance, save_target: base.save_target.clone() },
                &mut out,
            );
        }
    }
    out.truncate(k.max(1));
    out
}

fn explored_enemy_capital(state: &GameState, pov: PlayerId) -> Option<i32> {
    for (id, t) in &state.tribes {
        if *id == pov {
            continue;
        }
        for c in &t.cities {
            let Some(tile) = state.tiles.get(&c.idx) else { continue };
            if tile.capital_of == *id && tile.explorers.contains(&pov) {
                return Some(c.idx);
            }
        }
    }
    None
}

/// Stage 1: shallow macro lookahead over candidate directives.
pub struct MacroLookaheadAgent<'a> {
    evaluator: &'a Evaluator,
    params: MacroParams,
    stance_commit: StanceCommit,
    archetype: ArchetypeState,
    counters: TurnCounters,
    /// (settings.turn, pov) of the current plan — replan when it changes.
    plan_key: Option<(i32, PlayerId)>,
    turn_goal: Option<MacroGoal>,
    /// Turns where the chosen directive differed from the scripted base.
    pub divergent_turns: u32,
    pub planned_turns: u32,
}

impl<'a> MacroLookaheadAgent<'a> {
    pub fn new(evaluator: &'a Evaluator, params: MacroParams) -> Self {
        Self {
            evaluator,
            params,
            stance_commit: StanceCommit::default(),
            archetype: ArchetypeState::default(),
            counters: TurnCounters::default(),
            plan_key: None,
            turn_goal: None,
            divergent_turns: 0,
            planned_turns: 0,
        }
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let key = (game.state.settings.turn, pov);
        if self.plan_key != Some(key) {
            self.replan(game, pov);
            self.plan_key = Some(key);
        }
        let goal = self.turn_goal.clone().unwrap_or_default();
        let mut view = game.clone_for_mcts(pov);
        let ranked =
            rank_view(&mut view, pov, &goal, &mut self.archetype, &mut self.counters, self.params.lambda);
        let m = first_true_legal(game, ranked);
        self.counters.count(m.as_ref());
        Some(m)
    }

    /// Roll every candidate out `horizon` own-turns (candidate goal on turn 1,
    /// fresh scripted goals after; opponent ghost-played between), score the
    /// leaves, commit the argmax for the whole turn. Ties keep the earlier
    /// candidate, so the scripted base wins ties.
    fn replan(&mut self, game: &Game, pov: PlayerId) {
        let view0 = game.clone_for_mcts(pov);
        // The turn's single StanceCommit advance — production hysteresis
        // semantics stay on the script track even when an override wins.
        let base = update_goal(&view0.state, pov, &mut self.stance_commit, self.counters.tier3_bought);
        let candidates =
            enumerate_candidates(&view0.state, pov, base.clone(), self.counters, self.params.k);

        let mut scores: Vec<Option<f32>> = vec![None; candidates.len()];
        let mut net_batch: Vec<(usize, crate::ai::features::RawFeatures)> = Vec::new();

        for (i, cand) in candidates.iter().enumerate() {
            let mut sim = view0.clone();
            let mut arch = self.archetype.clone();
            let mut counters = self.counters;
            for h in 0..self.params.horizon.max(1) {
                let goal_h = if h == 0 {
                    cand.clone()
                } else {
                    scripted_goal(&sim.state, pov, counters.tier3_bought)
                };
                if !macro_exec::execute_turn(&mut sim, pov, &goal_h, &mut arch, &mut counters, self.params.lambda)
                    || sim.state.settings._game_over
                    || !macro_exec::ghost_until(&mut sim, pov)
                    || sim.state.settings._game_over
                {
                    break;
                }
            }

            if sim.state.settings._game_over {
                let my = sim.state.tribes.get(&pov).map(|t| t.score).unwrap_or(0);
                let opp = sim
                    .state
                    .tribes
                    .iter()
                    .filter(|(id, _)| **id != pov)
                    .map(|(_, t)| t.score)
                    .max()
                    .unwrap_or(0);
                scores[i] = Some(if my > opp {
                    1.0
                } else if my < opp {
                    -1.0
                } else {
                    0.0
                });
                continue;
            }
            match self.params.leaf {
                MacroLeaf::Heuristic => {
                    scores[i] = Some(crate::ai::evaluate_state(&sim.state, pov));
                }
                MacroLeaf::Net => {
                    // The candidate goal is painted into the leaf features, so
                    // win_value scores "this state, pursuing this goal".
                    match crate::ai::features::state_to_cpu_features_goal(
                        &sim.state,
                        pov,
                        None,
                        Some(cand),
                    ) {
                        Ok(f) => net_batch.push((i, f)),
                        Err(_) => scores[i] = Some(crate::ai::evaluate_state(&sim.state, pov)),
                    }
                }
            }
        }

        if !net_batch.is_empty() {
            let (idxs, feats): (Vec<usize>, Vec<_>) = net_batch.into_iter().unzip();
            // EvalResult.0 (win_value) only — .1 is stubbed 0.0 on tch/metal.
            for (i, r) in idxs.into_iter().zip(self.evaluator.evaluate(feats)) {
                scores[i] = Some(r.0);
            }
        }

        let mut best = 0usize;
        for (i, s) in scores.iter().enumerate() {
            if s.unwrap_or(f32::NEG_INFINITY) > scores[best].unwrap_or(f32::NEG_INFINITY) {
                best = i;
            }
        }
        self.planned_turns += 1;
        if best != 0 {
            self.divergent_turns += 1;
        }
        self.turn_goal = Some(candidates.into_iter().nth(best).unwrap_or(base));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval_server::DummyEvalHandle;
    use crate::states::UnitState;
    use crate::types::UnitType;

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
    fn fogged_clone_honesty() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let opp: PlayerId = if pov == 1 { 2 } else { 1 };
            let size = game.state.settings.size as i32;
            // Farthest tile pov has never explored, away from its units.
            let anchor = game.state.tribes[&pov].units[0].coords.idx;
            let cheb = |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
            let hidden = (0..size * size)
                .filter(|idx| {
                    game.state.tiles.get(idx).map_or(false, |t| {
                        !t.explorers.contains(&pov)
                            && t.terrain_type == crate::types::TerrainType::Field
                    })
                })
                .max_by_key(|&idx| cheb(anchor, idx));
            let Some(hidden) = hidden else { continue };

            let mut spooked = game.clone();
            spooked.state.tribes.get_mut(&opp).unwrap().units.push(UnitState {
                unit_type: UnitType::Warrior,
                coords: crate::Coords::from_index(hidden, size),
                owner: opp,
                ..Default::default()
            });

            let m1 = MacroScriptAgent::new(1.0).select_move(&mut game.clone()).unwrap();
            let m2 = MacroScriptAgent::new(1.0).select_move(&mut spooked).unwrap();
            assert_eq!(
                m1.serialize().to_string(),
                m2.serialize().to_string(),
                "seed {seed}: an invisible enemy unit changed the plan"
            );
        }
    }

    #[test]
    fn candidates_sorted_deduped_base_first() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let base = scripted_goal(&game.state, pov, 0);
            let cands =
                enumerate_candidates(&game.state, pov, base.clone(), TurnCounters::default(), 6);
            assert!(!cands.is_empty() && cands.len() <= 6);
            assert_eq!(cands[0], base, "seed {seed}: base not first");
            for (i, c) in cands.iter().enumerate() {
                let mut sorted = c.orders.clone();
                sorted.sort();
                assert_eq!(c.orders, sorted, "seed {seed}: candidate {i} unsorted");
                assert!(
                    !cands[..i].contains(c),
                    "seed {seed}: duplicate candidate {i}"
                );
            }
        }
    }

    #[test]
    fn lookahead_returns_true_legal_move() {
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        for seed in 0..2i64 {
            let mut game = generated_game(seed);
            let mut agent = MacroLookaheadAgent::new(
                &evaluator,
                MacroParams { horizon: 1, ..Default::default() },
            );
            let m = agent.select_move(&mut game).unwrap();
            let legal: Vec<String> =
                game.legal_moves().iter().map(|x| x.serialize().to_string()).collect();
            assert!(
                legal.contains(&m.serialize().to_string()),
                "seed {seed}: chose a true-illegal move"
            );
        }
    }

    #[test]
    fn net_leaf_batches_candidates() {
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        let mut game = generated_game(0);
        let mut agent = MacroLookaheadAgent::new(
            &evaluator,
            MacroParams { leaf: MacroLeaf::Net, horizon: 1, ..Default::default() },
        );
        assert!(agent.select_move(&mut game).is_some());
        assert_eq!(agent.planned_turns, 1);
    }
}
