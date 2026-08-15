//! Execute one canonical replay and print either its final state or a contextual error.

use polyfish::replay::{ReplayExecutor, load_replay};

fn main() -> anyhow::Result<()> {
    let path = std::env::args_os().nth(1).ok_or_else(|| {
        anyhow::anyhow!("usage: cargo run --example replay_diag -- <replay.json>")
    })?;
    let replay = load_replay(&path)?;
    let game = ReplayExecutor::execute(&replay)?;
    println!(
        "valid: {} commands; final turn {}; active player {}",
        replay.command_count(),
        game.state.settings.turn,
        game.state.settings.current_player_turn_id,
    );
    Ok(())
}
