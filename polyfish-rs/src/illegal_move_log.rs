//! Diagnostic dump for illegal-move-attempt errors from `Game::simulate_move`
//! / `Game::play_move`. Complements the `SIM_MOVE_FAILURES` counter in
//! `game.rs` with a per-event JSON trace a developer can inspect offline.

use crate::moves::Move;
use crate::states::GameState;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_LOGGED_FAILURES: u64 = 500;

static LOGGED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Opt-in gate for simulated-move failure diagnostics (`POLYFISH_VERBOSE_SIM_FAILURES=1`),
/// covering both the `illegal_moves/*.json` dumps and game.rs's per-event stderr line.
/// These failures are expected, self-healed stale-tree-reuse events; dumping a full
/// `GameState` per event mid-search taxes self-play throughput and accumulates ~100MB
/// of debris for a non-bug. Real `play_move` rejections log unconditionally.
pub fn verbose_sim_failures() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("POLYFISH_VERBOSE_SIM_FAILURES").is_ok())
}

/// Dump one illegal-move-attempt event to `illegal_moves/<context>_<ts>_<id>.json`.
/// `state` is captured after the failed `execute()` call — safe today because
/// every known failing `Move::execute()` validates before mutating, but that's
/// an observed convention, not a guarantee for every `Move` impl.
/// Never panics: IO/serialize errors are eprintln'd and swallowed. Capped at
/// `MAX_LOGGED_FAILURES` files per process so a long, buggy run can't fill disk;
/// `SIM_MOVE_FAILURES` keeps counting past the cap regardless.
pub fn log_illegal_move(context: &str, state: &GameState, game_move: &dyn Move, error: &str) {
    let seen = LOGGED_COUNT.fetch_add(1, Ordering::Relaxed);
    if seen == MAX_LOGGED_FAILURES {
        eprintln!(
            "[illegal_move_log] reached {MAX_LOGGED_FAILURES} logged failures; \
             suppressing further illegal_moves/*.json writes (SIM_MOVE_FAILURES keeps counting)"
        );
    }
    if seen >= MAX_LOGGED_FAILURES {
        return;
    }

    let dir = std::path::Path::new("illegal_moves");
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("[illegal_move_log] failed to create illegal_moves/: {e}");
        return;
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let payload = serde_json::json!({
        "timestamp": timestamp,
        "context": context,
        "turn": state.settings.turn,
        "current_player_turn_id": state.settings.current_player_turn_id,
        "move_type": format!("{:?}", game_move.move_type()),
        "move_description": game_move.describe(state),
        "move_json": game_move.serialize(),
        "error": error,
        "state": state,
    });

    let path = dir.join(format!("{context}_{timestamp}_{seen}.json"));
    match serde_json::to_vec_pretty(&payload) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("[illegal_move_log] failed to write {}: {e}", path.display());
            }
        }
        Err(e) => eprintln!("[illegal_move_log] failed to serialize illegal-move payload: {e}"),
    }
}
