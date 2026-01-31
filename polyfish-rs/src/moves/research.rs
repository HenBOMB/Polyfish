//! Research move
//!
//! Research a new technology.

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{MoveType, TechnologyType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchMove {
    pub tech: TechnologyType,
}

impl ResearchMove {
    pub fn new(tech: TechnologyType) -> Self {
        Self { tech }
    }
}

// ... (struct remains)

impl Move for ResearchMove {
    fn move_type(&self) -> MoveType {
        MoveType::Research
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;

        // Validation
        if let Some(tribe) = state.tribes.get(&pov_id) {
            let tech_cost = crate::functions::get_tech_cost(tribe, self.tech);
            if tribe.stars < tech_cost {
                return Err(format!(
                    "Insufficient stars for research: need {}, have {}",
                    tech_cost, tribe.stars
                ));
            }
        } else {
            return Err("Tribe not found".to_string());
        }

        // Logic
        let undo = crate::actions::tech::unlock_tech(state, self.tech, false)?;
        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Research {:?}", self.tech)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Generate research moves
pub fn generate_research_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        // Collect available techs based on what we have
        let mut available_techs = std::collections::HashSet::new();

        // Always include Unrequired's children if we have Unrequired (everyone has it implicitly or explicitly)
        // Actually, just iterate all discovered techs and add their 'next'
        for tech_state in &tribe.tech_vanilla {
            // Get settings for this tech
            let settings =
                crate::settings::technology::get_technology_setting(tech_state.tech_type);
            for next_tech in settings.next {
                available_techs.insert(next_tech);
            }
        }

        // Also check if we have Unrequired explicitly? Usually yes.
        // If tech_vanilla is empty (impossible), add Unrequired?
        // Let's assume tech_vanilla is populated.

        for tech in available_techs {
            // Check if already discovered
            if tribe.tech_vanilla.iter().any(|t| t.tech_type == tech) {
                continue;
            }

            // Check cost
            let cost = crate::functions::get_tech_cost(tribe, tech);
            if tribe.stars >= cost {
                moves.push(Box::new(ResearchMove::new(tech)));
            }
        }
    }
}
