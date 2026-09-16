use crate::ai::macro_exec;
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{LaneState, MacroGoal, compute_macro_goal};
use crate::ai::search::macro_mcts::MacroMctsStats;
use crate::game::Game;
use crate::states::PlayerId;

// EXP_ELO_047 Phase A: JSONL row per planned root, env-gated (`POLYFISH_PAINT_PROBE=<path>`).
// Compares goal evaluations under alternate plans and checks antisymmetry (Phase B).
pub(super) fn paint_probe(
    eval: &crate::ai::eval_server::Evaluator,
    state: &crate::states::GameState,
    pov: PlayerId,
    scripted: &MacroGoal,
    committed: Option<&MacroGoal>,
    diverged: bool,
    stats: &MacroMctsStats,
) {
    let Ok(path) = std::env::var("POLYFISH_PAINT_PROBE") else {
        return;
    };
    let v = |p: PlayerId, g: &MacroGoal| -> Option<f32> {
        crate::ai::features::state_to_cpu_features_goal(state, p, None, Some(g))
            .ok()
            .and_then(|f| eval.evaluate(vec![f]).first().map(|r| r.0))
    };
    let opp: PlayerId = if pov == 1 { 2 } else { 1 };
    let opp_scripted = compute_macro_goal(state, opp, 0);
    let (Some(v_scripted), Some(v_committed), Some(v_opp)) = (
        v(pov, scripted),
        v(pov, committed.unwrap_or(scripted)),
        v(opp, &opp_scripted),
    ) else {
        return;
    };
    // Control for P3: the heuristic on the SAME (fogged) states. Fog alone
    // makes a view non-zero-sum for any evaluator, so the net's asymmetry is
    // only the net's insofar as it exceeds this.
    let h_pov = crate::ai::evaluate_state(state, pov);
    let h_opp = crate::ai::evaluate_state(state, opp);
    let f = |x: Option<f32>| x.map(|n| format!("{n:.5}")).unwrap_or("null".into());
    let row = format!(
        "{{\"turn\":{},\"pov\":{},\"diverged\":{},\"v_scripted\":{:.5},\"v_committed\":{:.5},\"v_opp\":{:.5},\"h_pov\":{:.5},\"h_opp\":{:.5},\"q_spread\":{},\"q_best\":{}}}\n",
        state.settings.turn,
        pov,
        diverged,
        v_scripted,
        v_committed,
        v_opp,
        h_pov,
        h_opp,
        f(stats.root_q_spread),
        f(stats.root_q),
    );
    use std::io::Write;
    if let Ok(mut fh) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = fh.write_all(row.as_bytes());
    }
}

/// EXP_ELO_048: does Tier 3 track Tier 2? Runs turn three ways (picked, base, none) on clones.
/// Compares actual executed ply sequences; env-enabled via `POLYFISH_TIER_PROBE=<path>`.
pub(super) fn tier_probe(
    view: &Game,
    pov: PlayerId,
    lane_state: &LaneState,
    counters: TurnCounters,
    lambda: f32,
    base: &MacroGoal,
    picked: &MacroGoal,
    diverged: bool,
) {
    let Ok(path) = std::env::var("POLYFISH_TIER_PROBE") else {
        return;
    };
    let run = |goal: &MacroGoal| -> (Vec<crate::ai::macro_exec::PlyRec>, Game) {
        let mut g = view.clone();
        let mut a = lane_state.clone();
        let mut c = counters;
        let mut rec = Vec::new();
        macro_exec::execute_turn_recorded(
            &mut g,
            pov,
            goal,
            &mut a,
            &mut c,
            lambda,
            Some(&mut rec),
        );
        (rec, g)
    };
    let (a, after_pick) = run(picked);
    let (b, _) = run(base);
    let (c, after_none) = run(&MacroGoal::default());
    // Multiset overlap on serialized moves: order-insensitive, so a reordered
    // but identical set of plies still reads as "the directive changed
    // nothing", which is the conservative direction for this question.
    let overlap =
        |x: &[crate::ai::macro_exec::PlyRec], y: &[crate::ai::macro_exec::PlyRec]| -> f32 {
            let mut pool: Vec<&String> = y.iter().map(|p| &p.mv).collect();
            let mut hit = 0usize;
            for p in x {
                if let Some(i) = pool.iter().position(|q| **q == p.mv) {
                    pool.remove(i);
                    hit += 1;
                }
            }
            let denom = x.len().max(y.len());
            if denom == 0 {
                1.0
            } else {
                hit as f32 / denom as f32
            }
        };
    // Only account for Research, Build, Summon plies (actual star spend).
    let spend = |v: &[crate::ai::macro_exec::PlyRec]| -> Vec<crate::ai::macro_exec::PlyRec> {
        v.iter()
            .filter(|p| matches!(p.kind.as_str(), "Research" | "Build" | "Summon"))
            .cloned()
            .collect()
    };
    let (sa, sb, sc) = (spend(&a), spend(&b), spend(&c));
    let flips_phi = a.iter().filter(|p| p.flip_no_phi).count();
    let flips_goal = a.iter().filter(|p| p.flip_no_goal).count();
    // Obedience test: for each order, compare nearest unit's pre/post distance to target with and without the directive.
    // If distance closes equally without the order, it wasn't truly followed—just coincidental.
    let size = view.state.settings.size;
    let dist = |state: &crate::states::GameState, target: i32| -> Option<i32> {
        let t = crate::coords::Coords::from_index(target, size);
        state
            .tribes
            .get(&pov)?
            .units
            .iter()
            .map(|u| u.coords.chebyshev_distance_to(&t))
            .min()
    };
    let owned = |state: &crate::states::GameState, target: i32| -> bool {
        state
            .tiles
            .get(&target)
            .map_or(false, |t| t.owner == pov as i32)
    };
    let orders: Vec<String> = picked
        .orders
        .iter()
        .map(|(kind, t)| {
            let f = |x: Option<i32>| x.map(|v| v.to_string()).unwrap_or("null".into());
            format!(
                "{{\"kind\":\"{kind:?}\",\"target\":{t},\"d_pre\":{},\"d_pick\":{},\"d_none\":{},\
\"owned_pre\":{},\"owned_pick\":{}}}",
                f(dist(&view.state, *t)),
                f(dist(&after_pick.state, *t)),
                f(dist(&after_none.state, *t)),
                owned(&view.state, *t),
                owned(&after_pick.state, *t),
            )
        })
        .collect();
    let row = format!(
        "{{\"turn\":{},\"pov\":{},\"diverged\":{},\"plies\":{},\"spend_plies\":{},\
\"overlap_pick_base\":{:.4},\"overlap_pick_none\":{:.4},\
\"spend_overlap_pick_base\":{:.4},\"spend_overlap_pick_none\":{:.4},\
\"flip_no_phi\":{},\"flip_no_goal\":{},\"orders\":[{}]}}\n",
        view.state.settings.turn,
        pov,
        diverged,
        a.len(),
        sa.len(),
        overlap(&a, &b),
        overlap(&a, &c),
        overlap(&sa, &sb),
        overlap(&sa, &sc),
        flips_phi,
        flips_goal,
        orders.join(","),
    );
    use std::io::Write;
    if let Ok(mut fh) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = fh.write_all(row.as_bytes());
    }
}
