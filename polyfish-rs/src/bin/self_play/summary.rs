//! The end-of-run console summary: wall-clock breakdown, throughput, and
//! the global diagnostic counters various experiments left behind.
//!
//! `Throughput: N moves/sec` is a stdout contract (bench_eval_sweep.sh).

use std::time::{Duration, Instant};

/// Prints the run breakdown and flushes the dphi probe. Reads a pile of
/// global atomics, so it takes almost nothing as parameters.
pub(crate) fn print_run_summary(
    start_time: Instant,
    games_duration: Duration,
    total_moves: usize,
) {
    let total_duration = start_time.elapsed();
    println!("\n=== Self-Play Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f32());
    println!("Breakdown:");
    println!(
        "  - Game generation: {:.2}s ({:.1}%)",
        games_duration.as_secs_f32(),
        100.0 * games_duration.as_secs_f32() / total_duration.as_secs_f32()
    );
    let final_moves_per_sec = total_moves as f64 / games_duration.as_secs_f64().max(1e-9);
    println!(
        "  - Throughput: {:.2} moves/sec ({} moves)",
        final_moves_per_sec, total_moves
    );
    // How often search crossed a turn boundary in-tree (simulated EndTurn
    // edges only; real played moves don't count). ~0/move decision means the
    // tree essentially never sees beyond the current turn.
    let sim_end_turns =
        polyfish::game::SIM_END_TURN_EDGES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim EndTurn edges: {} total ({:.2} per move decision)",
        sim_end_turns,
        sim_end_turns as f64 / (total_moves as f64).max(1.0)
    );
    // How often a simulated move failed to execute against the replayed state
    // (tree-reuse staleness in Gumbel MCTS — see SIM_MOVE_FAILURES doc comment
    // in game.rs). Set POLYFISH_VERBOSE_SIM_FAILURES=1 for illegal_moves/*.json dumps.
    let sim_move_failures =
        polyfish::game::SIM_MOVE_FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim move failures: {} total ({:.2} per move decision)",
        sim_move_failures,
        sim_move_failures as f64 / (total_moves as f64).max(1.0)
    );
    // Ply-distillation throughput envelope input (EXP_ELO_061 GPU-ply-work
    // plan, Phase 0): how many rank_plies calls (rollout + real-commit)
    // and candidate moves per real move decision under macro-mcts. Zero
    // under gumbel (rank_plies is macro-mcts-only).
    let rank_plies_calls =
        polyfish::ai::search::macro_exec::RANK_PLIES_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let rank_plies_candidates = polyfish::ai::search::macro_exec::RANK_PLIES_CANDIDATES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - rank_plies calls: {} total ({:.2} per move decision), {} candidates total ({:.1} per call)",
        rank_plies_calls,
        rank_plies_calls as f64 / (total_moves as f64).max(1.0),
        rank_plies_candidates,
        rank_plies_candidates as f64 / (rank_plies_calls as f64).max(1.0)
    );
    println!(
        "  - EXP_ELO_083 tech-limit no-recommendation rejections (diagnostic, temporary): {} candidates",
        polyfish::ai::search::goal_aux::TECH_LIMIT_REJECTIONS.load(std::sync::atomic::Ordering::Relaxed)
    );
    if let Ok(m) = polyfish::ai::search::goal_aux::TECH_LIMIT_REJECTIONS_BY_TECH.lock() {
        let mut by_tech: Vec<(&polyfish::types::TechnologyType, &u64)> = m.iter().collect();
        by_tech.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> =
            by_tech.iter().take(12).map(|(t, c)| format!("{t:?}:{c}")).collect();
        println!("  - EXP_ELO_088 rejections by tech (diagnostic, top 12): {}", top.join(", "));
    }
    {
        let eligible = polyfish::ai::search::macro_exec::ENDTURN_ELIGIBLE_PLIES
            .load(std::sync::atomic::Ordering::Relaxed);
        let chosen = polyfish::ai::search::macro_exec::ENDTURN_CHOSEN_WITH_ALTERNATIVES
            .load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  - EXP_ELO_085 EndTurn chosen despite alternatives (diagnostic, temporary): {chosen}/{eligible} ({:.3}%)",
            100.0 * chosen as f64 / (eligible as f64).max(1.0)
        );
    }
    println!(
        "  - EXP_ELO_095 shared-attacker partial weights (diagnostic, temporary): {} entries",
        polyfish::ai::combat::SHARED_ATTACKER_PARTIAL_WEIGHTS.load(std::sync::atomic::Ordering::Relaxed)
    );
    {
        let candidates = polyfish::ai::search::macro_exec::STEP_LETHAL_ENTRY_CANDIDATES
            .load(std::sync::atomic::Ordering::Relaxed);
        let fires = polyfish::ai::search::macro_exec::STEP_LETHAL_ENTRY_FIRES
            .load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  - EXP_ELO_111 Step lethal-entry gate (diagnostic, temporary): {fires}/{candidates} ({:.3}%)",
            100.0 * fires as f64 / (candidates as f64).max(1.0)
        );
    }
    {
        let cover_total =
            polyfish::ai::combat::DEFEND_CREDIT_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let cover_partial =
            polyfish::ai::combat::DEFEND_CREDIT_PARTIAL.load(std::sync::atomic::Ordering::Relaxed);
        let hold_total =
            polyfish::ai::combat::DEFEND_HOLD_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let hold_partial =
            polyfish::ai::combat::DEFEND_HOLD_PARTIAL.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  - EXP_ELO_096 defend_cover fractional credit (diagnostic, temporary): {cover_partial}/{cover_total} ({:.3}%) assignments partial",
            100.0 * cover_partial as f64 / (cover_total as f64).max(1.0)
        );
        println!(
            "  - EXP_ELO_096 defend_hold fractional margin (diagnostic, temporary): {hold_partial}/{hold_total} ({:.3}%) evaluations partial",
            100.0 * hold_partial as f64 / (hold_total as f64).max(1.0)
        );
    }
    // Micro-mcts Phase 0 (throughput/cache-hit probe, POLYFISH_MICRO_PROBE_SIMS):
    // zero unless that env var is set. Note the rank_plies numbers above also
    // inflate while this probe is active -- its own continuation walk calls
    // rank_view/rank_plies, so that's the probe's real CPU cost showing up in
    // already-existing instrumentation, not contamination to filter out.
    let micro_probe_evals =
        polyfish::ai::search::macro_mcts::MICRO_PROBE_EVALS.load(std::sync::atomic::Ordering::Relaxed);
    let micro_probe_failures = polyfish::ai::search::macro_mcts::MICRO_PROBE_SIM_FAILURES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-probe evals: {} total ({:.2} per move decision), {} sim failures",
        micro_probe_evals,
        micro_probe_evals as f64 / (total_moves as f64).max(1.0),
        micro_probe_failures
    );
    let micro_mcts_calls =
        polyfish::ai::search::micro_mcts::MICRO_MCTS_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let micro_mcts_overrides = polyfish::ai::search::micro_mcts::MICRO_MCTS_OVERRIDES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-mcts calls: {} total, {} overrode rank_view's top pick ({:.1}%)",
        micro_mcts_calls,
        micro_mcts_overrides,
        micro_mcts_overrides as f64 / (micro_mcts_calls as f64).max(1.0) * 100.0
    );
    let micro_carry_attempts = polyfish::ai::search::micro_mcts::MICRO_CARRY_ATTEMPTS
        .load(std::sync::atomic::Ordering::Relaxed);
    let micro_carry_hits =
        polyfish::ai::search::micro_mcts::MICRO_CARRY_HITS.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-mcts root-advancement: {} carries offered, {} candidate children spliced in ({:.2} avg per carry)",
        micro_carry_attempts,
        micro_carry_hits,
        micro_carry_hits as f64 / (micro_carry_attempts as f64).max(1.0)
    );
    polyfish::ai::search::macro_exec::dphi_probe_flush();
}
