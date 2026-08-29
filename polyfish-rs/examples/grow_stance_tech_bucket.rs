//! EXP_ELO_080 (diagnostic): enumerate every TechnologyType's `arms`/`grows`
//! classification (mirroring `passes_stance_tech_mask`'s exact logic) to
//! find the "neither" bucket -- techs that pass freely, ungated, under
//! Grow stance despite not helping eco growth at all. Read-only.

use polyfish::settings::technology::{get_tech_effects, is_eco_tech};
use polyfish::types::TechnologyType;
use strum::IntoEnumIterator;

fn main() {
    let mut neither = Vec::new();
    let mut arms = Vec::new();
    let mut grows = Vec::new();
    let mut both = Vec::new();

    for tech in TechnologyType::iter() {
        if matches!(tech, TechnologyType::BeyondComprehension | TechnologyType::Basic) {
            continue;
        }
        let effects = get_tech_effects(tech);
        let a = !effects.combat_units.is_empty();
        let g = is_eco_tech(tech);
        match (a, g) {
            (true, true) => both.push(tech),
            (true, false) => arms.push(tech),
            (false, true) => grows.push(tech),
            (false, false) => neither.push(tech),
        }
    }

    println!("=== arms && grows (never gated under Grow, always eco+combat) ===");
    for t in &both {
        println!("  {t:?}");
    }
    println!("=== arms && !grows (GATED under Grow behind 5-star reserve) ===");
    for t in &arms {
        println!("  {t:?}");
    }
    println!("=== !arms && grows (passes freely under Grow -- correctly, it's eco) ===");
    for t in &grows {
        println!("  {t:?}");
    }
    println!("=== !arms && !grows (passes FREELY under Grow -- neither combat nor eco) ===");
    for t in &neither {
        println!("  {t:?}");
    }
    println!(
        "\ncounts: both={} arms_only={} grows_only={} neither={}",
        both.len(),
        arms.len(),
        grows.len(),
        neither.len()
    );
}
