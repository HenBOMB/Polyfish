use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::StepMove;
use polyfish::recorder::GameRecorder;
use polyfish::types::{MapSize, MapType, TribeType};
use std::fs;

#[test]
fn test_recorder_save() {
    // 1. Setup Game State
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny;
    settings.tribes = vec![TribeType::Imperius, TribeType::Oumaji];
    settings.seed = 12345;
    settings.map_type = MapType::Drylands;

    let initial_state = generate(settings);
    let mut game = Game::new();
    game.state = initial_state;
    game.post_load();

    // 2. Initialize Recorder
    let recorder = GameRecorder::new();

    // 3. Create a dummy move
    let step_move = StepMove::new(0, 1); // Indices might be invalid effectively but structure is valid

    // 4. Record Step
    // We pass arbitrary eco/mil values
    recorder.record_step(&game.state, &step_move, 0.5, 0.5);

    // 5. Attach an outcome; `save` refuses to write steps with no win label
    let finished = recorder.finish_game(Some(game.state.settings.current_player_turn_id));
    assert_eq!(finished, 1, "the recorded step must receive an outcome");

    // 6. Save
    let result = recorder.save();
    assert!(
        result.is_ok(),
        "Failed to save recorder data: {:?}",
        result.err()
    );

    let message = result.unwrap();
    println!("{}", message);

    // Parse filename from message "Saved 1 samples to human_games_X.safetensors"
    let parts: Vec<&str> = message.split_whitespace().collect();
    let filename = parts.last().unwrap(); // "human_games_X.safetensors"

    // 7. Verify file exists
    assert!(
        fs::metadata(filename).is_ok(),
        "File {} was not created",
        filename
    );

    // 8. Cleanup
    fs::remove_file(filename).expect("Failed to delete test file");
}
