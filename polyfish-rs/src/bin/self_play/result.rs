//! What one finished game hands back to the aggregator: the per-decision
//! record the shard is built from, the end-of-game metric bundle, and the
//! replay regrouping. No policy or label logic -- see labels.rs for that.

use polyfish::ai::features::{self, GameFeatures};
use polyfish::ai::mapper::DecomposedMapper;
use polyfish::replayer::{ModReplay, ReplayPlayer, ReplayTurn};
use polyfish::states::PlayerId;
use crate::tempo::TempoTrack;
use std::collections::HashMap;

/// Decomposed policy probability distributions for a single step
pub(crate) struct DecomposedPolicyData {
    pub(crate) action_type: Vec<f32>,    // [11]
    pub(crate) source_spatial: Vec<f32>, // [H * W]
    pub(crate) target_spatial: Vec<f32>, // [H * W]
    pub(crate) move_option: Vec<f32>,    // [192]
}

/// One recorded decision point. `my_score`/`opp_score`/`turn` are snapshotted
/// BEFORE this step's move executes. `root_value` is that same pre-move
/// state's post-search root value (see `GumbelMctsAgent::last_root_value`) —
/// the TD bootstrap target used by whichever *earlier* step's label lands on
/// this step as its "next decision" horizon.
pub(crate) struct HistoryStep {
    pub(crate) features: GameFeatures,
    pub(crate) policy: DecomposedPolicyData,
    pub(crate) player_id: PlayerId,
    pub(crate) my_score: f32,
    pub(crate) opp_score: f32,
    pub(crate) turn: i32,
    pub(crate) root_value: Option<f32>,
    /// Raw NN root value (tanh-bounded, pre-search) — value-head calibration only.
    pub(crate) root_own_value: Option<f32>,
    /// Ground-truth (unfogged) non-invisible enemy-unit occupancy at decision
    /// time, POV-relative — the aux_fog_units target.
    pub(crate) enemy_units: Vec<f32>,
    pub(crate) my_spt: i32,
    pub(crate) opp_spt: i32,
    /// `(city tile, production)` for every city the POV holds at decision time
    /// — the raw material for the aux_city_spt target.
    pub(crate) city_spt: Vec<(i32, i32)>,
    /// Pursuit proximity to the nearest capturable village at decision time,
    /// POV-relative, normalized to [0,1] — the aux_pursuit target.
    pub(crate) pursuit: f32,
    /// EXP_ELO_061 (Stage 3b): the macro root's own candidate ballot and
    /// post-search visit counts, captured once per (turn, pov) via
    /// `macro_ballot_for_history_step` — `None` on every ply after the
    /// first within a turn (the ballot is stable all turn; capturing it on
    /// every ply would just duplicate the same target across each ply's
    /// distinct feature vector), or when this seat isn't running
    /// macro-mcts. Raw material for the macro_stance/macro_order targets;
    /// marginalized in post-game processing, not here.
    pub(crate) macro_ballot: Option<(Vec<polyfish::ai::oracle_macro::MacroGoal>, Vec<f32>)>,
}

/// Result from a single game - contains all data needed for training
pub(crate) struct GameResult {
    pub(crate) history: Vec<HistoryStep>,
    pub(crate) scores: HashMap<i32, i32>,
    /// Per-player `score + shape_w_label·Φ` at game end — the terminal
    /// snapshot for TD labels, consistent with the shaped step snapshots.
    /// Equals raw score when shaping is off.
    pub(crate) final_potentials: HashMap<i32, f32>,
    pub(crate) final_cities: HashMap<i32, i32>,
    pub(crate) total_cities: i32,
    pub(crate) moves: usize,
    /// Net-seat plies only (excludes Greedy/opponent seats) — the seat-clean
    /// counterpart of `moves` for the avg_moves behavior chart.
    pub(crate) net_moves: usize,
    pub(crate) winner_score: i32,
    /// Adjudicated winner: sole survivor, else higher final score at timeout.
    pub(crate) winner_id: i32,
    pub(crate) recap: ModReplay,
    pub(crate) cap_ruins: usize,
    pub(crate) cap_villages: usize,
    pub(crate) cap_cities: usize,
    pub(crate) cap_capitals: usize,
    pub(crate) action_counts: HashMap<polyfish::types::MoveType, usize>,
    /// Move-type counts keyed by turn number, for the "move mix by turn"
    /// training-progress chart (see parse_metrics.py / dashboard).
    pub(crate) moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>>,
    /// NET-seat tile-exploration and territory-ownership counts at game end
    /// (anchor/opponent seats excluded since Jul 2026; in mirror self-play
    /// this still sums both seats).
    pub(crate) revealed_tiles: i32,
    pub(crate) captured_tiles: i32,
    /// Realized level of the adjacency hubs a net seat BUILT, as
    /// `(hubs, partner_sum, hubs_at_most_1, hubs_lost)` per structure type.
    /// `max_affordable_pop` prices a hub at its BEST placement, so this is the
    /// planned-vs-delivered pop gap; a hub at 1 partner costs 5★ for 1 pop,
    /// worse than the LumberHut feeding it. Attribution is by builder, not by
    /// end-of-game tile owner — the latter credits captured anchor hubs.
    pub(crate) hub_levels: HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)>,
    /// First hub of each type the net built, as
    /// `(partners_chosen, partners_best, sites_that_beat_it, sites_available,
    /// terrain_ceiling_chosen, terrain_ceiling_best)`. The first pair is scored
    /// on hubs actually built (so it inherits the net's hut policy); the ceiling
    /// pair is terrain+resource only, i.e. the site's potential.
    pub(crate) first_hub_rank: HashMap<polyfish::types::StructureType, (i64, i64, u32, u32, i64, i64)>,
    /// Turn by which 50%/80%/100% of the map's initial open villages (and
    /// ruins) had been captured by a NET-controlled seat — how *directly*
    /// the net seeks them out. Censored at max_turns when a game never gets
    /// there (incl. when the anchor takes them first — losing the race
    /// reads as censored, not captured).
    pub(crate) villages_t2c_p50: f32,
    pub(crate) villages_t2c_p80: f32,
    pub(crate) villages_t2c_all: f32,
    /// First-village stats, per NET SEAT (2 in a mirror game, 1 in an
    /// anchor/league game) so the aggregator can divide by seats rather than
    /// games — matching the t2c_Nth_rate family. `censored_sum` charges
    /// max_turns to a seat that never captured; `turn_sum` covers only the
    /// seats that did.
    pub(crate) villages_first_seats: u32,
    pub(crate) villages_first_captured: u32,
    pub(crate) villages_first_turn_sum: f64,
    pub(crate) villages_first_censored_sum: f64,
    pub(crate) ruins_t2c_p50: f32,
    pub(crate) ruins_t2c_p80: f32,
    pub(crate) ruins_t2c_all: f32,
    /// Mean tribe SPT sampled at the start of game turns 0, 5, 10, … (player 1
    /// to act, before any moves on that turn).
    pub(crate) spt_at_turn: HashMap<i32, f32>,
    /// (mean unit worth, mean army stars per city) over net seats, at the same
    /// milestones as `spt_at_turn`. Absolute ratios with no opponent term, so
    /// unlike contested counts they can move in mirror self-play; measured
    /// cv ~1.5%/iteration against a Greedy reference of ~3.7 / ~10.0 at t15.
    pub(crate) army_ratios_at_turn: HashMap<i32, (f32, f32)>,
    /// End-of-game ground truth for the aux heads: raw per-tile owner ids,
    /// per-player SPT, and per-player researched-tech multi-hot.
    pub(crate) final_owner: Vec<i32>,
    pub(crate) final_spt: HashMap<PlayerId, i32>,
    pub(crate) final_tech: HashMap<PlayerId, Vec<f32>>,
    /// Per-player tempo curves + unit-accounting counters.
    pub(crate) tempo: HashMap<PlayerId, TempoTrack>,
    /// Seat roles (index = player_id - 1): "model", "model_vs_anchor",
    /// "anchor", or "opponent" — lets the aggregator split tempo curves into
    /// intrinsic (mirror), contested (vs anchor), and reference populations.
    pub(crate) roles: [&'static str; 2],
}

/// Aggregate a move-visit distribution into the four decomposed policy-target
/// arrays (action / source-spatial / target-spatial / option), each normalized
/// to sum 1. Shared by the MCTS visit target and the EXP_ELO_020 DAgger
/// Greedy-teacher target so both are built identically before blending.
pub(crate) fn decompose_visits(
    move_visits: &[polyfish::ai::mcts_types::MoveVisit],
    map_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let spatial = features::MAP_SIZE * features::MAP_SIZE;
    let mut p_action = vec![0.0; 11];
    let mut p_source = vec![0.0; spatial];
    let mut p_target = vec![0.0; spatial];
    let mut p_option = vec![0.0; 192];
    let mut total = 0.0;
    for mv in move_visits {
        total += mv.visits;
        let t = DecomposedMapper::move_visit_to_targets(mv, map_size);
        if t.action_type < p_action.len() {
            p_action[t.action_type] += mv.visits;
        }
        if let Some(i) = t.source_spatial {
            if i < p_source.len() {
                p_source[i] += mv.visits;
            }
        }
        if let Some(i) = t.target_spatial {
            if i < p_target.len() {
                p_target[i] += mv.visits;
            }
        }
        if let Some(i) = t.target_type {
            if i < p_option.len() {
                p_option[i] += mv.visits;
            }
        }
    }
    if total > 0.0 {
        for x in &mut p_action {
            *x /= total;
        }
        for x in &mut p_source {
            *x /= total;
        }
        for x in &mut p_target {
            *x /= total;
        }
        for x in &mut p_option {
            *x /= total;
        }
    }
    (p_action, p_source, p_target, p_option)
}

pub(crate) fn group_recap(flat: Vec<(i32, i32, serde_json::Value)>) -> Vec<ReplayTurn> {
    let mut turns: Vec<ReplayTurn> = Vec::new();
    for (turn_num, player_id, cmd) in flat {
        if turns.is_empty() || turns.last().unwrap().turn != turn_num {
            turns.push(ReplayTurn {
                turn: turn_num,
                players: Vec::new(),
            });
        }
        let turn = turns.last_mut().unwrap();
        if turn.players.is_empty() || turn.players.last().unwrap().player_id != player_id {
            turn.players.push(ReplayPlayer {
                player_id,
                commands: Vec::new(),
            });
        }
        turn.players.last_mut().unwrap().commands.push(cmd);
    }
    turns
}
