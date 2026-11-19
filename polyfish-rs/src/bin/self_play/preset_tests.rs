use super::*;
use polyfish::research::OpponentPreset;

#[test]
fn preset_games_store_only_the_learner_seat_but_replay_both() {
    let network = PolyZeroNet::new(candle_nn::VarBuilder::zeros(
        candle_core::DType::F32,
        &candle_core::Device::Cpu,
    ))
    .unwrap();
    let evaluator = Evaluator::Dummy(polyfish::ai::eval_server::DummyEvalHandle::new());
    let aborted = AtomicUsize::new(0);
    let traces = AtomicUsize::new(0);
    for (index, bot_seat) in [(0, 2), (1, 1)] {
        let result = play_single_game(
            &network,
            &network,
            &evaluator,
            &evaluator,
            1,
            index,
            42,
            vec![TribeType::Imperius, TribeType::Bardur],
            1,
            usize::MAX,
            2,
            Some((OpponentPreset::Normal, bot_seat)),
            SearchBackend::Greedy,
            SearchBackend::Greedy,
            None,
            None,
            ProgressMode::SampledFinish,
            false,
            0,
            false,
            TraceTrigger::Adjacent,
            0,
            &traces,
            &aborted,
        )
        .unwrap();
        assert!(result.anchor_seat.is_none());
        assert!(!result.history.is_empty());
        assert!(result.history.iter().all(|step| step.player_id != bot_seat));
        assert!(result.history.len() < result.recap.command_count());
        assert!(
            result
                .recap
                .turns
                .iter()
                .any(|turn| turn.player_id == bot_seat)
        );
        assert_eq!(
            result
                .recap
                .initial_state
                .settings
                .research
                .unwrap()
                .bot_seat,
            bot_seat
        );
    }
    assert_eq!(aborted.load(Ordering::Relaxed), 0);
}
