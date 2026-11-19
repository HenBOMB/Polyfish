use std::{fs::File, path::PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use polyfish::game::Game;
use polyfish::replay::{
    ReplayMetadata, ReplayPlayerMetadata, ReplayRecorder, ReplayResult, ReplaySource,
};
use polyfish::research::{self, MatchConfig, MatchResult, OpponentPreset};
use serde_json::json;

#[derive(Parser)]
#[command(about = "Play a seeded, model-free Polyfish research match")]
struct Args {
    #[arg(long, default_value_t = 121)]
    map_size: u16,
    #[arg(long, default_value = "drylands", value_parser = ["drylands", "lakes", "continents", "pangea", "archipelago", "water-world"])]
    map_type: String,
    #[arg(long, default_value_t = 42)]
    seed: i64,
    #[arg(long, default_value = "imperius")]
    tribe1: String,
    #[arg(long, default_value = "bardur")]
    tribe2: String,
    #[arg(long, value_enum, default_value = "normal")]
    preset1: OpponentPreset,
    #[arg(long, value_enum, default_value = "normal")]
    preset2: OpponentPreset,
    #[arg(long, default_value_t = research::DEFAULT_MAX_TURNS)]
    max_turns: i32,
    #[arg(long)]
    replay: Option<PathBuf>,
}

fn record(game: &Game, args: &Args) -> ReplayRecorder {
    ReplayRecorder::new(
        game.state.clone(),
        ReplayMetadata {
            source: ReplaySource::Other,
            game_id: None,
            created_at: None,
            map_width: game.map_size() as usize,
            map_height: game.map_size() as usize,
            max_turns: game.state.settings.max_turns,
            game_mode: game.state.settings.mode,
            players: game
                .state
                .tribes
                .values()
                .map(|tribe| ReplayPlayerMetadata {
                    player_id: tribe.id,
                    tribe: tribe.tribe_type,
                    name: None,
                })
                .collect(),
            source_diagnostics: Some(json!({
                "presetVersion": research::PRESET_VERSION,
                "presets": [args.preset1, args.preset2],
                "seed": args.seed,
                "mapType": args.map_type,
            })),
        },
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let size = research::board_size(args.map_size)
        .context("Choose 121, 196, 256, 324, 400, or 900 tiles")?;
    let kind = research::map_type(&args.map_type).context("Unsupported map type")?;
    let tribes = [
        research::research_tribe(&args.tribe1).context("Unsupported player-one tribe")?,
        research::research_tribe(&args.tribe2).context("Unsupported player-two tribe")?,
    ];
    if args.max_turns < 1 {
        bail!("The turn limit must be positive");
    }
    let mut game = research::new_match_on(
        size,
        kind,
        tribes,
        args.seed,
        MatchConfig::new(args.preset2.into(), 2),
    );
    game.state.settings.max_turns = args.max_turns;
    let mut recorder = args.replay.as_ref().map(|_| record(&game, &args));
    let mut actions = 0;
    while research::result(&game) == MatchResult::InProgress {
        let seat = game.current_player_id();
        let preset = if seat == 1 {
            args.preset1
        } else {
            args.preset2
        };
        let turn = game.turn();
        let decision = game.decision();
        let move_ = research::choose_from_decision(&decision, preset)
            .context("Bot did not select a legal move")?;
        let action = polyfish::Action::from_move(move_.as_ref())?;
        let _undo = decision
            .play(&action)
            .context("Bot move rejected by the engine")?;
        if let Some(recorder) = &mut recorder {
            recorder.record_command(turn, seat, action)?;
        }
        actions += 1;
    }
    let result = research::result(&game);
    if let (Some(path), Some(recorder)) = (args.replay, recorder) {
        let winner_player_id = match result {
            MatchResult::Won { winner } => Some(winner),
            _ => None,
        };
        let replay = recorder.finish(Some(ReplayResult {
            winner_player_id,
            draw: false,
            scores: game
                .state
                .tribes
                .values()
                .map(|tribe| (tribe.id, tribe.score))
                .collect(),
            reason: Some(
                if winner_player_id.is_some() {
                    "mightVictory"
                } else {
                    "turnLimit"
                }
                .into(),
            ),
        }));
        serde_json::to_writer_pretty(File::create(path)?, &replay)?;
    }
    println!(
        "{}",
        json!({
            "result": result,
            "seed": args.seed,
            "mapSize": args.map_size,
            "mapType": args.map_type,
            "turn": game.turn(),
            "actions": actions,
        })
    );
    Ok(())
}
