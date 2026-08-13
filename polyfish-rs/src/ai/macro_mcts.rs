//! EXP_ELO_033: adversarial turn-level MCTS over macro directives (Stage 2 of
//! the macro-search redesign). Nodes are turn boundaries, edges are
//! `MacroGoal` directives executed by the deterministic executor, and — the
//! upgrade over the Stage-1 lookahead — the OPPONENT's turns are searched
//! adversarially instead of ghost-scripted. Two-player only; negamax backup
//! over the antisymmetric heuristic `evaluate_state`.

use crate::ai::macro_agent::{MacroParams, enumerate_candidates};
use crate::ai::macro_exec::{self, TurnCounters};
use crate::ai::oracle_macro::{ArchetypeState, MacroGoal, StanceCommit, scripted_goal, update_goal};
use crate::game::Game;
use crate::moves::Move;
use crate::states::{GameState, PlayerId};

/// UCT exploration on [0,1]-mapped values, dialed to the MEASURED root q01
/// spread between directives (0.01–0.06, smoke probe 2026-08-12): at c=0.6
/// (and even the HeuristicMctsAgent 0.6 precedent) the bonus is 6–30x the
/// signal and visits stay uniform. 0.05 makes a 0.03 q01 gap decisive within
/// ~10 visits while genuine ties still split evenly.
const EXPLORATION: f32 = 0.05;

/// Tree depth cap in game turns from the root — beyond it a node is scored by
/// the leaf evaluator instead of expanded.
const TURN_DEPTH_CAP: i32 = 8;

fn other(p: PlayerId) -> PlayerId {
    if p == 1 { 2 } else { 1 }
}

fn seat(p: PlayerId) -> usize {
    (p != 1) as usize
}

/// Approximate a seat's whole-game purchase counters from the (fogged) state:
/// techs discovered after turn 0 (starting techs — Basic + the tribe tech —
/// are stamped turn 0 by mapgen; ruin grants are counted too, an acceptable
/// over-count). Keeps the tech caps binding for the in-tree opponent instead
/// of letting it out-research reality.
pub fn derive_counters(state: &GameState, player: PlayerId) -> TurnCounters {
    let Some(tribe) = state.tribes.get(&player) else {
        return TurnCounters::default();
    };
    let bought: Vec<_> = tribe
        .tech_vanilla
        .iter()
        .filter(|t| t.discovered && t.discovered_turn > 0)
        .collect();
    let tier3 = bought
        .iter()
        .filter(|t| {
            crate::settings::technology::get_technology_setting(t.tech_type).tier == Some(3)
        })
        .count() as u32;
    TurnCounters { techs_bought: bought.len() as u32, tier3_bought: tier3 }
}

/// Terminal value from `perspective`'s side by score comparison — the same
/// convention the backup expects of `evaluate_state` at that node.
fn terminal_value(state: &GameState, perspective: PlayerId) -> f32 {
    let my = state.tribes.get(&perspective).map(|t| t.score).unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != perspective)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);
    if my > opp {
        1.0
    } else if my < opp {
        -1.0
    } else {
        0.0
    }
}

/// One tree node = one turn boundary. `player` is the acting player BY
/// ALTERNATION from the root (not read from the state — a game that ends
/// mid-edge leaves `current_player_turn_id` unreliable). Every value stored
/// or computed at a node is from `player`'s perspective; the backup negates
/// once per level (negamax).
struct Node {
    game: Game,
    player: PlayerId,
    counters: [TurnCounters; 2],
    arch: [ArchetypeState; 2],
    candidates: Vec<MacroGoal>,
    children: Vec<Option<usize>>,
    edge_visits: Vec<f32>,
    edge_values: Vec<f32>,
    visits: f32,
    /// Terminal (game over) or depth-capped leaf value, from `player`'s
    /// perspective; computed once (the executor and evaluator are
    /// deterministic within a process).
    frozen_value: Option<f32>,
}

impl Node {
    fn new(
        game: Game,
        player: PlayerId,
        counters: [TurnCounters; 2],
        arch: [ArchetypeState; 2],
        root_turn: i32,
        k: usize,
    ) -> Self {
        let frozen_value = if game.state.settings._game_over {
            Some(terminal_value(&game.state, player))
        } else if game.state.settings.turn - root_turn >= TURN_DEPTH_CAP {
            Some(crate::ai::evaluate_state(&game.state, player))
        } else {
            None
        };
        let candidates = if frozen_value.is_some() {
            Vec::new()
        } else {
            let base = scripted_goal(&game.state, player, counters[seat(player)].tier3_bought);
            enumerate_candidates(&game.state, player, base, counters[seat(player)], k)
        };
        let n = candidates.len();
        Node {
            game,
            player,
            counters,
            arch,
            candidates,
            children: vec![None; n],
            edge_visits: vec![0.0; n],
            edge_values: vec![0.0; n],
            visits: 0.0,
            frozen_value,
        }
    }

    /// UCT over edges on [0,1]-mapped Q; unvisited edges first, in candidate
    /// order (base first).
    fn select_edge(&self) -> usize {
        if let Some(i) = self.edge_visits.iter().position(|&v| v == 0.0) {
            return i;
        }
        let ln_n = self.visits.max(1.0).ln();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..self.candidates.len() {
            let q01 = (self.edge_values[i] / self.edge_visits[i] + 1.0) / 2.0;
            let score = q01 + EXPLORATION * (ln_n / self.edge_visits[i]).sqrt();
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}

/// Search telemetry from the last `run` call (smoke instrumentation).
#[derive(Clone, Copy, Debug, Default)]
pub struct MacroMctsStats {
    pub nodes: usize,
    pub max_depth: usize,
    pub root_visit_max_share: f32,
}

pub struct MacroMctsSearch {
    nodes: Vec<Node>,
    pub stats: MacroMctsStats,
}

impl MacroMctsSearch {
    /// Run `sims` simulations from `root_game` (the acting player's fogged
    /// view) and return the winning root directive index. Root candidate 0
    /// must be the committed script base; ties break toward it.
    pub fn run(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_arch: &ArchetypeState,
        params: &MacroParams,
    ) -> (usize, MacroMctsStats) {
        Self::run_with(root_game, pov, root_candidates, own_counters, own_arch, params, |_| {})
    }

    fn run_with(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_arch: &ArchetypeState,
        params: &MacroParams,
        inspect: impl FnOnce(&MacroMctsSearch),
    ) -> (usize, MacroMctsStats) {
        debug_assert_eq!(root_game.state.tribes.len(), 2, "macro MCTS is 2-player only");
        let root_turn = root_game.state.settings.turn;
        let mut counters = [TurnCounters::default(); 2];
        counters[seat(pov)] = own_counters;
        counters[seat(other(pov))] = derive_counters(&root_game.state, other(pov));
        let mut arch: [ArchetypeState; 2] = Default::default();
        arch[seat(pov)] = own_arch.clone();

        let mut root = Node::new(root_game.clone(), pov, counters, arch, root_turn, params.k);
        root.candidates = root_candidates;
        let n = root.candidates.len();
        root.children = vec![None; n];
        root.edge_visits = vec![0.0; n];
        root.edge_values = vec![0.0; n];

        let mut search = MacroMctsSearch { nodes: vec![root], stats: MacroMctsStats::default() };
        for _ in 0..params.sims.max(1) {
            search.simulate(0, root_turn, params);
        }
        inspect(&search);

        let root = &search.nodes[0];
        let mut best = 0;
        for i in 1..root.edge_visits.len() {
            if root.edge_visits[i] > root.edge_visits[best] {
                best = i;
            }
        }
        search.stats.nodes = search.nodes.len();
        search.stats.root_visit_max_share = if root.visits > 0.0 {
            root.edge_visits.iter().cloned().fold(0.0, f32::max) / root.visits
        } else {
            0.0
        };
        (best, search.stats)
    }

    /// `run` plus a per-edge root dump on stdout (smoke instrumentation only).
    #[cfg(test)]
    pub fn run_probed(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_arch: &ArchetypeState,
        params: &MacroParams,
    ) -> (usize, MacroMctsStats) {
        let cands_dbg: Vec<String> = root_candidates
            .iter()
            .map(|c| format!("{:?}/{}ord", c.stance, c.orders.len()))
            .collect();
        let (best, stats) = Self::run_with(root_game, pov, root_candidates, own_counters, own_arch, params, |s| {
            let root = &s.nodes[0];
            for i in 0..root.candidates.len() {
                let q = if root.edge_visits[i] > 0.0 {
                    root.edge_values[i] / root.edge_visits[i]
                } else {
                    f32::NAN
                };
                println!("    edge {i} [{}]: visits={} q={q:+.4}", cands_dbg[i], root.edge_visits[i]);
            }
        });
        (best, stats)
    }

    /// One descent: walk UCT edges until an unexpanded edge, expand it (one
    /// `execute_turn` on a fresh clone), then back the child's value up the
    /// path with one negation per level.
    fn simulate(&mut self, root_idx: usize, root_turn: i32, params: &MacroParams) {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut idx = root_idx;
        let mut value: f32;
        loop {
            if let Some(v) = self.nodes[idx].frozen_value {
                value = v;
                break;
            }
            if self.nodes[idx].candidates.is_empty() {
                value = crate::ai::evaluate_state(&self.nodes[idx].game.state, self.nodes[idx].player);
                break;
            }
            let e = self.nodes[idx].select_edge();
            path.push((idx, e));
            if let Some(child) = self.nodes[idx].children[e] {
                idx = child;
                continue;
            }
            let child = self.expand(idx, e, root_turn, params);
            let cn = &self.nodes[child];
            value = cn
                .frozen_value
                .unwrap_or_else(|| crate::ai::evaluate_state(&cn.game.state, cn.player));
            // The child's value is from the child's perspective; the edge we
            // just descended belongs to the parent, so negate once here and
            // once per level in the unwind below.
            break;
        }
        for &(pidx, e) in path.iter().rev() {
            value = -value;
            let node = &mut self.nodes[pidx];
            node.visits += 1.0;
            node.edge_visits[e] += 1.0;
            node.edge_values[e] += value;
        }
        self.stats.max_depth = self.stats.max_depth.max(path.len());
    }

    fn expand(&mut self, parent: usize, edge: usize, root_turn: i32, params: &MacroParams) -> usize {
        let (mut game, player, mut counters, mut arch, goal) = {
            let p = &self.nodes[parent];
            (
                p.game.clone(),
                p.player,
                p.counters,
                p.arch.clone(),
                p.candidates[edge].clone(),
            )
        };
        let s = seat(player);
        // An executor anomaly leaves the state where it stopped; the node is
        // still scoreable, so treat it like any other boundary.
        let _ = macro_exec::execute_turn(
            &mut game,
            player,
            &goal,
            &mut arch[s],
            &mut counters[s],
            params.lambda,
        );
        let child = Node::new(game, other(player), counters, arch, root_turn, params.k);
        let child_idx = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent].children[edge] = Some(child_idx);
        child_idx
    }
}

/// Stage 2 agent: per-turn directive commit like the Stage-1 lookahead, but
/// the directive is chosen by the adversarial turn-level tree.
pub struct MacroMctsAgent {
    params: MacroParams,
    stance_commit: StanceCommit,
    archetype: ArchetypeState,
    counters: TurnCounters,
    plan_key: Option<(i32, PlayerId)>,
    turn_goal: Option<MacroGoal>,
    pub divergent_turns: u32,
    pub planned_turns: u32,
    pub last_stats: MacroMctsStats,
}

impl MacroMctsAgent {
    pub fn new(params: MacroParams) -> Self {
        Self {
            params,
            stance_commit: StanceCommit::default(),
            archetype: ArchetypeState::default(),
            counters: TurnCounters::default(),
            plan_key: None,
            turn_goal: None,
            divergent_turns: 0,
            planned_turns: 0,
            last_stats: MacroMctsStats::default(),
        }
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let key = (game.state.settings.turn, pov);
        if self.plan_key != Some(key) {
            let view0 = game.clone_for_mcts(pov);
            let base =
                update_goal(&view0.state, pov, &mut self.stance_commit, self.counters.tier3_bought);
            let candidates =
                enumerate_candidates(&view0.state, pov, base.clone(), self.counters, self.params.k);
            let (pick, stats) = MacroMctsSearch::run(
                &view0,
                pov,
                candidates.clone(),
                self.counters,
                &self.archetype,
                &self.params,
            );
            self.last_stats = stats;
            self.planned_turns += 1;
            if pick != 0 {
                self.divergent_turns += 1;
            }
            self.turn_goal = candidates.into_iter().nth(pick);
            self.plan_key = Some(key);
        }
        let goal = self.turn_goal.clone().unwrap_or_default();
        let mut view = game.clone_for_mcts(pov);
        let ranked = crate::ai::macro_agent::rank_view(
            &mut view,
            pov,
            &goal,
            &mut self.archetype,
            &mut self.counters,
            self.params.lambda,
        );
        let m = crate::ai::macro_agent::first_true_legal(game, ranked);
        self.counters.count(m.as_ref());
        Some(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Negamax correctness depends on this: evaluate_state must be
    /// antisymmetric in a 2-player game, including after several executed
    /// turns of drift.
    #[test]
    fn evaluate_state_is_antisymmetric() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            for _ in 0..6 {
                let s = evaluate_asym(&sim.state);
                assert!(s < 1e-4, "seed {seed}: antisymmetry violated by {s}");
                if sim.state.settings._game_over {
                    break;
                }
                let player = sim.state.settings.current_player_turn_id;
                let goal = scripted_goal(&sim.state, player, 0);
                let mut arch = ArchetypeState::default();
                let mut counters = TurnCounters::default();
                if !macro_exec::execute_turn(&mut sim, player, &goal, &mut arch, &mut counters, 1.0)
                {
                    break;
                }
            }
        }
    }

    fn evaluate_asym(state: &GameState) -> f32 {
        (crate::ai::evaluate_state(state, 1) + crate::ai::evaluate_state(state, 2)).abs()
    }

    #[test]
    fn tree_goes_deeper_than_the_root() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let view = game.clone_for_mcts(pov);
            let base = scripted_goal(&view.state, pov, 0);
            let cands = enumerate_candidates(&view.state, pov, base, TurnCounters::default(), 4);
            let k = cands.len();
            let (_, stats) = MacroMctsSearch::run(
                &view,
                pov,
                cands,
                TurnCounters::default(),
                &ArchetypeState::default(),
                &MacroParams { sims: 32, ..Default::default() },
            );
            assert!(
                stats.nodes > k + 1,
                "seed {seed}: only {} nodes for k={k} at 32 sims — no second-level expansion",
                stats.nodes
            );
            assert!(stats.max_depth >= 2, "seed {seed}: max depth {} < 2", stats.max_depth);
        }
    }

    #[test]
    fn mcts_agent_returns_true_legal_move() {
        for seed in 0..2i64 {
            let mut game = generated_game(seed);
            let mut agent = MacroMctsAgent::new(MacroParams { sims: 16, ..Default::default() });
            let m = agent.select_move(&mut game).unwrap();
            let legal: Vec<String> =
                game.legal_moves().iter().map(|x| x.serialize().to_string()).collect();
            assert!(legal.contains(&m.serialize().to_string()), "seed {seed}: true-illegal move");
        }
    }

    /// EXP_ELO_033 smoke probe (manual): advances each game to mid-game with
    /// the executor, then prints per-edge root Q/visits so tuning can tell
    /// genuine directive ties from an exploration term swamping the signal.
    /// Run: cargo test --lib ai::macro_mcts -- --ignored --nocapture
    #[test]
    #[ignore]
    fn smoke_stats_probe() {
        for seed in 0..8i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let mut arch: [ArchetypeState; 2] = Default::default();
            let mut counters = [TurnCounters::default(); 2];
            for _ in 0..8 {
                if sim.state.settings._game_over {
                    break;
                }
                let p = sim.state.settings.current_player_turn_id;
                let goal = scripted_goal(&sim.state, p, counters[seat(p)].tier3_bought);
                if !macro_exec::execute_turn(
                    &mut sim,
                    p,
                    &goal,
                    &mut arch[seat(p)],
                    &mut counters[seat(p)],
                    1.0,
                ) {
                    break;
                }
            }
            if sim.state.settings._game_over {
                continue;
            }
            let pov = sim.state.settings.current_player_turn_id;
            let base = scripted_goal(&sim.state, pov, counters[seat(pov)].tier3_bought);
            let cands =
                enumerate_candidates(&sim.state, pov, base, counters[seat(pov)], 4);
            let k = cands.len();
            let sims = std::env::var("PROBE_SIMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            let t0 = std::time::Instant::now();
            let (pick, stats) = MacroMctsSearch::run_probed(
                &sim,
                pov,
                cands,
                counters[seat(pov)],
                &arch[seat(pov)],
                &MacroParams { sims, ..Default::default() },
            );
            println!(
                "seed {seed} t{}: k={k} pick={pick} nodes={} depth={} share={:.2} ({}ms)",
                sim.state.settings.turn,
                stats.nodes,
                stats.max_depth,
                stats.root_visit_max_share,
                t0.elapsed().as_millis()
            );
        }
    }

    #[test]
    fn derived_counters_track_discovered_techs() {
        let game = generated_game(0);
        for pid in [1, 2] {
            let c = derive_counters(&game.state, pid);
            assert_eq!(c.techs_bought, 0, "fresh game should derive 0 bought techs");
            assert_eq!(c.tier3_bought, 0);
        }
    }
}
