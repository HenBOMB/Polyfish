use polyfish::ai::evaluator::economy::evaluate_economy;
use polyfish::states::{CityState, GameState, TribeState};

fn main() {
    println!("Comparing Economy Evaluators (New vs Old Logic) with Base Income");
    println!("---------------------------------------------------------------");
    println!(
        "{:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Stars", "New Score", "Old Score", "Diff", "Penalty"
    );
    println!("{:-<65}", "-");

    // We need a dummy state
    let mut state = GameState::default();
    let player_id = 1;
    let mut tribe = TribeState::default();
    tribe.id = player_id;

    // Give some base income so we can see the penalty subtract from it
    // Income comes from cities. Let's make a dummy city.
    // production=10 means income = 10 + 1 = 11.
    let mut city = CityState::default();
    city.production = 10;
    tribe.cities.push(city);

    // iterate stars from 0 to 20
    for stars in 0..=20 {
        tribe.stars = stars;
        state.tribes.insert(player_id, tribe.clone());

        // New Score (Actual function call)
        let new_score = evaluate_economy(&state, player_id);

        // Old Logic Simulation (Approximate)
        // Income score = (11 / 25).clamp(0,1) = 0.44
        // Weighted Income = 0.44 * 0.5 = 0.22
        // Stars score = (stars / 25).clamp(0,1)
        // Weighted Stars = stars_score * 0.2
        // Tech = 0

        let income_score = (11.0 / 25.0f32).clamp(0.0, 1.0);
        let weighted_income = income_score * 0.5; // 0.22

        let stars_score_old = (stars as f32 / 25.0).clamp(0.0, 1.0);
        let weighted_stars = stars_score_old * 0.2;

        // Old Score (ignoring other penalties for now, assuming 0)
        let old_score = weighted_income + weighted_stars;

        let diff = new_score - old_score;

        // Calculate expected penalty for validation
        let threshold = 8.0;
        let penalty = if (stars as f32) < threshold {
            let deficit = 1.0 - (stars as f32 / threshold);
            deficit.powi(2) * 0.25
        } else {
            0.0
        };

        println!(
            "{:5} | {:10.4} | {:10.4} | {:10.4} | {:10.4}",
            stars, new_score, old_score, diff, penalty
        );
    }

    println!("\nHigh Income Scenario (100 SPT)");
    println!("--------------------------------");
    println!(
        "{:>5} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Stars", "New Score", "Old Score", "Diff", "Penalty"
    );
    println!("{:-<65}", "-");

    // Scenario 2: High Income
    let mut high_income_tribe = TribeState::default();
    high_income_tribe.id = player_id;
    let mut city = CityState::default();
    city.production = 100; // 100 SPT + 1 base = 101 income
    high_income_tribe.cities.push(city);

    for stars in 0..=10 {
        high_income_tribe.stars = stars;
        state.tribes.insert(player_id, high_income_tribe.clone());

        let new_score = evaluate_economy(&state, player_id);

        // Old Logic Simulation (High Income)
        // Income score = (101 / 25).clamp(0,1) = 1.0
        // Weighted Income = 1.0 * 0.5 = 0.5
        let income_score = 1.0;
        let weighted_income = 0.5;

        let stars_score_old = (stars as f32 / 25.0).clamp(0.0, 1.0);
        let weighted_stars = stars_score_old * 0.2;
        let old_score = weighted_income + weighted_stars;

        let diff = new_score - old_score;
        let penalty = if (stars as f32) < 8.0 {
            let deficit = 1.0 - (stars as f32 / 8.0);
            deficit.powi(2) * 0.25
        } else {
            0.0
        };

        println!(
            "{:5} | {:10.4} | {:10.4} | {:10.4} | {:10.4}",
            stars, new_score, old_score, diff, penalty
        );
    }
}
