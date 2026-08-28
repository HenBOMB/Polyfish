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
        if state.settings._verbose {
            state
                ._messages
                .push(format!("📚 Researched {:?}!", self.tech_type));
        }
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

    fn cost(&self, state: &GameState) -> Option<i32> {
        let tribe = state.tribes.get(&state.settings.current_player_turn_id)?;
        Some(crate::functions::get_tech_cost(tribe, self.tech_type))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::TribeState;
    use crate::types::TribeType;

    fn base_state() -> GameState {
        let mut state = GameState::default();
        state.settings.current_player_turn_id = 1;

        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.tribe_type = TribeType::Imperius;
        tribe.stars = 9999; // affordable regardless of cost curve
        state.tribes.insert(1, tribe);
        state
    }

    #[test]
    fn research_pushes_message_only_when_verbose() {
        let mut verbose_state = base_state();
        verbose_state.settings._verbose = true;
        let res = ResearchMove::new(TechnologyType::Riding).execute(&mut verbose_state);
        assert!(res.is_ok(), "research should succeed: {:?}", res.err());
        assert_eq!(
            verbose_state._messages,
            vec!["📚 Researched Riding!".to_string()],
            "expected exactly one research message when _verbose is true"
        );

        let mut quiet_state = base_state();
        // _verbose defaults to false
        let res = ResearchMove::new(TechnologyType::Riding).execute(&mut quiet_state);
        assert!(res.is_ok(), "research should succeed: {:?}", res.err());
        assert!(
            quiet_state._messages.is_empty(),
            "expected no message when _verbose is false, got {:?}",
            quiet_state._messages
        );
    }
}
