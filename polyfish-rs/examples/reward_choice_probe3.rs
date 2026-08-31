//! EXP_ELO_113: lightweight reward-choice comparator using the real
//! `score_move_with_unit_goals` heuristic score directly (not the fuller
//! `goal_potential_breakdown` decomposition `reward_choice_probe2.rs` does --
//! that probe has bit-rotted against newer signatures; this one stays
//! minimal on purpose). Prints both candidates' final scores and the winner.
//! Usage: cargo run --example reward_choice_probe3 -- <replay.json> <target_idx> <city_idx> <reward_a> <reward_b>
//! reward_a/reward_b are CityRewardType Debug names, e.g. BorderGrowth PopGrowth.
use polyfish::game::Game;
use polyfish::moves::RewardMove;
use polyfish::replayer::ModReplay;
use polyfish::types::CityRewardType;

fn parse_reward(s: &str) -> CityRewardType {
    match s {
        "Explorer" => CityRewardType::Explorer,
        "Workshop" => CityRewardType::Workshop,
        "CityWall" => CityRewardType::CityWall,
        "Resources" => CityRewardType::Resources,
        "PopGrowth" => CityRewardType::PopGrowth,
        "BorderGrowth" => CityRewardType::BorderGrowth,
        "Park" => CityRewardType::Park,
        "SuperUnit" => CityRewardType::SuperUnit,
        other => panic!("unknown reward type {other}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let target_idx: usize = args[2].parse().unwrap();
    let city_idx: i32 = args[3].parse().unwrap();
    let reward_a = parse_reward(&args[4]);
    let reward_b = parse_reward(&args[5]);
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    'outer: for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    break 'outer;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    let mv_a = RewardMove::new(city_idx, reward_a);
    let mv_b = RewardMove::new(city_idx, reward_b);
    let score_a = polyfish::ai::scoring::score_move_with_unit_goals(&game, &mv_a, None, None);
    let score_b = polyfish::ai::scoring::score_move_with_unit_goals(&game, &mv_b, None, None);
    println!(
        "{reward_a:?} score={score_a:.3}  {reward_b:?} score={score_b:.3}  winner={}",
        if score_a > score_b { format!("{reward_a:?}") } else { format!("{reward_b:?}") }
    );
}
