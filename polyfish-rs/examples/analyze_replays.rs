//! Canonical replay analyser. Execution is delegated to the shared executor.

use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::replay::{
    ReplayCommand, ReplayError, ReplayExecutor, ReplayMoveContext, ReplayObserver, load_replay,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Default)]
struct Counts(BTreeMap<&'static str, usize>);

impl ReplayObserver for Counts {
    fn before_move(
        &mut self,
        _game: &Game,
        _context: &ReplayMoveContext,
        _legal_moves: &[Box<dyn Move>],
        _selected_move: &dyn Move,
        command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        let name = match command {
            ReplayCommand::Step { .. } => "step",
            ReplayCommand::Attack { .. } => "attack",
            ReplayCommand::Capture { .. } => "capture",
            ReplayCommand::Build { .. } => "build",
            ReplayCommand::Research { .. } => "research",
            ReplayCommand::Summon { .. } => "summon",
            ReplayCommand::Upgrade { .. } => "upgrade",
            ReplayCommand::Ability { .. } => "ability",
            ReplayCommand::Reward { .. } => "reward",
            ReplayCommand::Harvest { .. } => "harvest",
            ReplayCommand::EndTurn => "endTurn",
            ReplayCommand::Resign => "resign",
        };
        *self.0.entry(name).or_default() += 1;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let paths: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    anyhow::ensure!(
        !paths.is_empty(),
        "usage: cargo run --example analyze_replays -- <replay.json>..."
    );
    for path in paths {
        let replay = load_replay(&path)?;
        let mut counts = Counts::default();
        let final_game = ReplayExecutor::execute_with_observer(&replay, &mut counts)?;
        println!(
            "{}: {} commands, final turn {}, counts {:?}",
            path.display(),
            replay.command_count(),
            final_game.state.settings.turn,
            counts.0,
        );
    }
    Ok(())
}
