//! Research move
//!
//! Research a new technology.

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{MoveType, TechnologyType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResearchMove {
    pub tech_type: TechnologyType,
}

impl ResearchMove {
    pub fn new(tech: TechnologyType) -> Self {
        Self { tech_type: tech }
    }
}

impl Move for ResearchMove {
    fn move_type(&self) -> MoveType {
        MoveType::Research
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;

        // Validation
        if let Some(tribe) = state.tribes.get(&pov_id) {
            let tech_cost = crate::functions::get_tech_cost(tribe, self.tech_type);
            if tribe.stars < tech_cost {
                return Err(format!(
                    "Insufficient stars for research: need {} (Cities: {}), have {}",
                    tech_cost,
                    tribe.cities.len(),
                    tribe.stars
                ));
            }
        } else {
            return Err("Tribe not found".to_string());
        }

        // Logic
        let undo = crate::actions::tech::unlock_tech(state, self.tech_type, false)?;
        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Research {:?}", self.tech_type)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.tech_type,
        })
    }

    #[inline]
    fn tech_type(&self) -> Result<TechnologyType, String> {
        Ok(self.tech_type)
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
            for &next_tech in &settings.next {
                let resolved = crate::settings::technology::resolve_tech_for_tribe(
                    next_tech,
                    tribe.tribe_type,
                );
                available_techs.insert(resolved);
            }

            // Backward: also allow researching the prerequisite of this tech
            let mut req_opt = settings.requires;

            // If checking a replacement tech that doesn't explicitly override requires, check the replaced tech
            if req_opt.is_none() {
                if let Some(replaced) = settings.replaces_tech {
                    let replaced_settings =
                        crate::settings::technology::get_technology_setting(replaced);
                    req_opt = replaced_settings.requires;
                }
            }

            if let Some(req_tech) = req_opt {
                let resolved =
                    crate::settings::technology::resolve_tech_for_tribe(req_tech, tribe.tribe_type);
                available_techs.insert(resolved);
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
