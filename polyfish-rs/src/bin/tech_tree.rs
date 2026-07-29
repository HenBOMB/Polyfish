//! Dump the annotated tech tree as JSON (one object per tech) for the
//! Python-side labelers/analysis (EXP_ELO_028 UNLOCK groundwork).
//!
//! Usage: cargo run --bin tech_tree > tech_tree.json

use polyfish::settings::technology::{
    get_tech_effects, get_technology_setting, is_eco_tech, is_military_tech, is_mobility_tech,
};
use polyfish::types::TechnologyType;
use strum::IntoEnumIterator;

fn main() {
    let mut out = Vec::new();
    for t in TechnologyType::iter() {
        let s = get_technology_setting(t);
        let e = get_tech_effects(t);
        out.push(serde_json::json!({
            "tech": format!("{t:?}"),
            "tier": s.tier,
            "requires": s.requires.map(|r| format!("{r:?}")),
            "replaces": s.replaces_tech.map(|r| format!("{r:?}")),
            "tribe": s.tribe_type.map(|r| format!("{r:?}")),
            "military": is_military_tech(t),
            "eco": is_eco_tech(t),
            "mobility": is_mobility_tech(t),
            "combat_units": e.combat_units.iter().map(|u| format!("{u:?}")).collect::<Vec<_>>(),
            "support_units": e.support_units.iter().map(|u| format!("{u:?}")).collect::<Vec<_>>(),
            "special_units": e.special_units.iter().map(|u| format!("{u:?}")).collect::<Vec<_>>(),
            "defense_bonus_terrain": e.defense_bonus_terrain.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "harvests": e.harvests.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
            "eco_structures": e.eco_structures.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "score_structures": e.score_structures.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "mobility_terrain": e.mobility_terrain.map(|x| format!("{x:?}")),
            "mobility_structures": e.mobility_structures.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "other_structures": e.other_structures.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "abilities": e.abilities.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "tasks": e.tasks.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>(),
            "capital_vision": e.capital_vision,
            "tech_discount": e.tech_discount,
        }));
    }
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
