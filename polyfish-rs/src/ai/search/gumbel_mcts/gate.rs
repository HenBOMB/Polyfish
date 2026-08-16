//! Root-only move gates (star gate / tech caps / ability gate / capture-
//! first) and their attribution stats (Aug 2026 taxonomy split out of
//! gumbel_mcts.rs to keep every file under ~1000 lines). `gate_stats` is
//! re-exported through `gumbel_mcts` so `crate::ai::gumbel_mcts::gate_stats`
//! (self_play's `POLYFISH_DUMP_GATE_BLOCKS` dump) keeps resolving.

use crate::moves::Move;
use crate::types::MoveType;


/// `POLYFISH_REUSED_ROOT_GATES=0` restores the pre-Aug-2 behavior where root
/// gates applied only on fresh roots (~1 ply in 9). Default on; the off switch
/// exists to measure what the gate-leak fix costs in activity, since the gates
/// behind it were dialed while they were mostly inert.
pub(super) fn reused_root_gates_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("POLYFISH_REUSED_ROOT_GATES").as_deref() != Ok("0"))
}
/// Which root gate rejects `m`, or `None` if it passes. Single source of truth
/// for the three sites that filter root candidates.
///
/// Attribution is FIRST-BLOCKER-WINS: a move both gates would reject is
/// charged only to the earlier one, so per-gate counts are lower bounds on
/// what each gate would block in isolation.
pub(super) fn gate_block(
    state: &crate::states::GameState,
    m: &dyn Move,
    star_gate: bool,
    stance: Option<crate::ai::oracle_macro::Stance>,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> Option<usize> {
    if star_gate && !crate::ai::oracle_macro::passes_star_gate(state, m, stance, aux) {
        return Some(0);
    }
    if let Some(a) = aux {
        if !crate::ai::oracle_macro::passes_tech_caps(m, a) {
            return Some(1);
        }
        if !crate::ai::oracle_macro::passes_ability_gate(state, m) {
            return Some(2);
        }
        if !crate::ai::oracle_macro::passes_capture_first(state, m) {
            return Some(3);
        }
    }
    None
}
/// Retain predicate shared by both real gating sites, recording an attributed
/// block when `POLYFISH_DUMP_GATE_BLOCKS=1`. EndTurn is always exempt so the
/// root can never be emptied.
pub(super) fn gate_retain(
    state: &crate::states::GameState,
    m: &dyn Move,
    star_gate: bool,
    stance: Option<crate::ai::oracle_macro::Stance>,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> bool {
    if m.move_type() == MoveType::EndTurn {
        return true;
    }
    match gate_block(state, m, star_gate, stance, aux) {
        None => true,
        Some(g) => {
            gate_stats::record(g, m.move_type(), state.settings.turn);
            false
        }
    }
}
/// Per-gate attribution of blocked root candidates, off unless
/// `POLYFISH_DUMP_GATE_BLOCKS=1`. Lock-free counters: the gates run on every
/// actor thread, and this must not perturb the thing it measures.
pub mod gate_stats {
    use crate::types::MoveType;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const GATES: [&str; 4] = ["star_gate", "tech_caps", "ability_gate", "capture_first"];
    const BANDS: [&str; 4] = ["t1_5", "t6_10", "t11_15", "t16up"];
    const N_TYPES: usize = 12;

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("POLYFISH_DUMP_GATE_BLOCKS").as_deref() == Ok("1"))
    }

    fn counts() -> &'static Vec<AtomicU64> {
        static C: OnceLock<Vec<AtomicU64>> = OnceLock::new();
        C.get_or_init(|| {
            (0..GATES.len() * N_TYPES * BANDS.len())
                .map(|_| AtomicU64::new(0))
                .collect()
        })
    }

    fn totals() -> &'static Vec<AtomicU64> {
        static T: OnceLock<Vec<AtomicU64>> = OnceLock::new();
        // [plies seen, plies that lost >=1 candidate, candidates in, candidates out]
        T.get_or_init(|| (0..4).map(|_| AtomicU64::new(0)).collect())
    }

    fn band(turn: i32) -> usize {
        match turn {
            ..=5 => 0,
            6..=10 => 1,
            11..=15 => 2,
            _ => 3,
        }
    }

    pub fn record(gate: usize, mt: MoveType, turn: i32) {
        if !enabled() {
            return;
        }
        let t = (mt as usize).min(N_TYPES - 1);
        let idx = gate * (N_TYPES * BANDS.len()) + t * BANDS.len() + band(turn);
        counts()[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// One gated root: how many candidates went in and how many survived.
    pub fn record_ply(before: usize, after: usize) {
        if !enabled() {
            return;
        }
        let t = totals();
        t[0].fetch_add(1, Ordering::Relaxed);
        if after < before {
            t[1].fetch_add(1, Ordering::Relaxed);
        }
        t[2].fetch_add(before as u64, Ordering::Relaxed);
        t[3].fetch_add(after as u64, Ordering::Relaxed);
    }

    pub fn snapshot() -> serde_json::Value {
        if !enabled() {
            return serde_json::Value::Null;
        }
        let c = counts();
        let mut gates = serde_json::Map::new();
        for (gi, gname) in GATES.iter().enumerate() {
            let mut by_type = serde_json::Map::new();
            for t in 0..N_TYPES {
                let mut by_band = serde_json::Map::new();
                let mut any = 0u64;
                for (bi, bname) in BANDS.iter().enumerate() {
                    let v = c[gi * (N_TYPES * BANDS.len()) + t * BANDS.len() + bi]
                        .load(Ordering::Relaxed);
                    any += v;
                    by_band.insert((*bname).to_string(), v.into());
                }
                if any > 0 {
                    by_band.insert("total".to_string(), any.into());
                    by_type.insert(
                        format!("{:?}", MoveType::from(t as i32)),
                        serde_json::Value::Object(by_band),
                    );
                }
            }
            gates.insert((*gname).to_string(), serde_json::Value::Object(by_type));
        }
        let t = totals();
        serde_json::json!({
            "by_gate": gates,
            "plies_gated": t[0].load(Ordering::Relaxed),
            "plies_losing_a_candidate": t[1].load(Ordering::Relaxed),
            "candidates_in": t[2].load(Ordering::Relaxed),
            "candidates_out": t[3].load(Ordering::Relaxed),
        })
    }
}
