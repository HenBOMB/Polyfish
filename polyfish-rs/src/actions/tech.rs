//! Technology actions

use crate::actions::{UndoCallback, spend_stars};
use crate::functions::get_tech_cost;
use crate::states::{GameState, TechnologyState};
use crate::types::TechnologyType;

/// Unlock a technology for the current tribe
pub fn unlock_tech(
    state: &mut GameState,
    tech_type: TechnologyType,
    free: bool,
) -> Result<UndoCallback, String> {
    let pov_id = state.settings.current_player_turn_id;
    let cost = if free {
        0
    } else {
        match state.tribes.get(&pov_id) {
            Some(tribe) => get_tech_cost(tribe, tech_type),
            None => 0,
        }
    };

    let mut undos = Vec::new();

    // Spend stars if not free
    if !free {
        undos.push(spend_stars(state, cost));
    }

    // Always `discovered`: real states only ever carry discovered techs, so a
    // simulated research must unlock the same units/structures/harvests a real
    // one does, or the search can never plan "research X, then use X".
    let tech_state = TechnologyState {
        tech_type,
        discovered: true,
        discovered_turn: state.settings.turn,
    };

    // Add tech to tribe
    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
        tribe.tech_vanilla.push(tech_state);

        // Price through `tech_tier`, not the raw `.tier`: replacement techs
        // carry no tier of their own and inherit their vanilla counterpart's,
        // so reading `.tier.unwrap_or(1)` here would disagree with the
        // canonical `calculate_detailed_tribe_score` (which uses `tech_tier`)
        // and drift `score_parity`.
        let score_gain = 100 * crate::settings::technology::tech_tier(tech_type);
        tribe.score += score_gain;

        undos.push(Box::new(move |s| {
            if let Some(t) = s.tribes.get_mut(&pov_id) {
                t.score -= score_gain;
                t.tech_vanilla.pop();
            }
        }));
    }

    use crate::actions::chain_undos;
    Ok(chain_undos(undos))
}
