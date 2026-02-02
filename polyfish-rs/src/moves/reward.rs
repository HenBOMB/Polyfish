//! Reward move (City Level Up Reward)

use crate::actions::city::claim_territory;
use crate::actions::discovery::{discover_tiles, predict_explorer};
use crate::actions::units::summon_unit;
use crate::actions::{chain_undos, gain_stars, UndoCallback};
use crate::functions::get_adjacent_indices;
use crate::moves::{Move, MoveResult};
use crate::settings::units::get_super_unit;
use crate::states::GameState;
use crate::types::{MoveType, RewardType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewardMove {
    /// City tile index
    pub target: i32,
    pub reward: RewardType,
}

impl RewardMove {
    pub fn new(target: i32, reward: RewardType) -> Self {
        Self { target, reward }
    }
}

impl Move for RewardMove {
    fn move_type(&self) -> MoveType {
        MoveType::Reward
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos: Vec<UndoCallback> = Vec::new();
        let target = self.target;
        let reward_type = self.reward;
        let p_id = state.settings.current_player_turn_id;

        // Find city
        let city_idx = if let Some(tribe) = state.tribes.get(&p_id) {
            tribe.cities.iter().position(|c| c.tile_index == target)
        } else {
            None
        };

        if let Some(c_idx) = city_idx {
            // Add to rewards set
            if let Some(tribe) = state.tribes.get_mut(&p_id) {
                if let Some(city) = tribe.cities.get_mut(c_idx) {
                    city.rewards.insert(reward_type);
                }
            }

            undos.push(Box::new(move |s: &mut GameState| {
                if let Some(t) = s.tribes.get_mut(&p_id) {
                    if let Some(c) = t.cities.get_mut(c_idx) {
                        c.rewards.remove(&reward_type);
                    }
                }
            }) as UndoCallback);

            match reward_type {
                RewardType::Workshop => {
                    if let Some(tribe) = state.tribes.get_mut(&p_id) {
                        if let Some(city) = tribe.cities.get_mut(c_idx) {
                            city.production += 1;
                            undos.push(Box::new(move |s: &mut GameState| {
                                if let Some(t) = s.tribes.get_mut(&p_id) {
                                    if let Some(c) = t.cities.get_mut(c_idx) {
                                        c.production -= 1;
                                    }
                                }
                            }) as UndoCallback);
                        }
                    }
                }
                RewardType::Explorer => {
                    let (_, predicted) = predict_explorer(state, target);
                    undos.push(discover_tiles(state, p_id, None, Some(predicted)));
                }
                RewardType::CityWall => {
                    if let Some(tribe) = state.tribes.get_mut(&p_id) {
                        if let Some(city) = tribe.cities.get_mut(c_idx) {
                            city._walls = true;
                            undos.push(Box::new(move |s: &mut GameState| {
                                if let Some(t) = s.tribes.get_mut(&p_id) {
                                    if let Some(c) = t.cities.get_mut(c_idx) {
                                        c._walls = false;
                                    }
                                }
                            }) as UndoCallback);
                        }
                    }
                }
                RewardType::Resources => {
                    undos.push(gain_stars(state, 5));
                }
                RewardType::PopGrowth => {
                    if let Some(tribe) = state.tribes.get_mut(&p_id) {
                        if let Some(city) = tribe.cities.get_mut(c_idx) {
                            city.population += 3;
                            city.progress += 3;
                            tribe.score += 15;
                            undos.push(Box::new(move |s: &mut GameState| {
                                if let Some(t) = s.tribes.get_mut(&p_id) {
                                    t.score -= 15;
                                    if let Some(c) = t.cities.get_mut(c_idx) {
                                        c.population -= 3;
                                        c.progress -= 3;
                                    }
                                }
                            }) as UndoCallback);
                        }
                    }
                }
                RewardType::BorderGrowth => {
                    if let Some(tribe) = state.tribes.get_mut(&p_id) {
                        if let Some(city) = tribe.cities.get_mut(c_idx) {
                            city.border_size += 1;
                            let adj = get_adjacent_indices(state, target, 2);
                            undos.push(claim_territory(state, &adj, target, false));
                            undos.push(Box::new(move |s: &mut GameState| {
                                if let Some(t) = s.tribes.get_mut(&p_id) {
                                    if let Some(c) = t.cities.get_mut(c_idx) {
                                        c.border_size -= 1;
                                    }
                                }
                            }) as UndoCallback);
                        }
                    }
                }
                RewardType::Park => {
                    if let Some(tribe) = state.tribes.get_mut(&p_id) {
                        if let Some(city) = tribe.cities.get_mut(c_idx) {
                            city.production += 1;
                            tribe.score += 250;
                            undos.push(Box::new(move |s: &mut GameState| {
                                if let Some(t) = s.tribes.get_mut(&p_id) {
                                    t.score -= 250;
                                    if let Some(c) = t.cities.get_mut(c_idx) {
                                        c.production -= 1;
                                    }
                                }
                            }) as UndoCallback);
                        }
                    }
                }
                RewardType::SuperUnit => {
                    let tribe_type = state.tribes.get(&p_id).map(|t| t.tribe_type).unwrap();
                    let unit_type = get_super_unit(tribe_type);
                    match summon_unit(state, unit_type, target, false, false) {
                        Ok(res) => undos.push(res.undo),
                        Err(_) => {}
                    }
                }
                _ => {}
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Choose reward {:?} for city at {}",
            self.reward, self.target
        )
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Reward,
            "target": self.target,
            "reward": self.reward
        })
    }
}

pub fn generate_reward_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        use crate::types::TribeType;
        let is_luxidoor = tribe.tribe_type == TribeType::Luxidoor;

        for city in &tribe.cities {
            let mut required = (city.level - 1) as usize;
            let mut offset = 0;

            // Luxidoor skips rewards for level 2 and 3 of their starting capital
            if is_luxidoor && city.tile_index == tribe.starting_tile_coords.idx {
                required = required.saturating_sub(2);
                offset = 2;
            }

            if city.level > 1 && city.rewards.len() < required {
                let slot = city.rewards.len() + 1 + offset;
                let (opt1, opt2) = match slot {
                    1 => (RewardType::Explorer, RewardType::Workshop),
                    2 => (RewardType::CityWall, RewardType::Resources),
                    3 => (RewardType::PopGrowth, RewardType::BorderGrowth),
                    _ => (RewardType::Park, RewardType::SuperUnit),
                };
                moves.push(Box::new(RewardMove::new(city.tile_index, opt1)));
                moves.push(Box::new(RewardMove::new(city.tile_index, opt2)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{CityState, TribeState};
    use crate::types::TribeType;

    #[test]
    fn test_luxidoor_no_initial_rewards() {
        let mut state = GameState::default();
        let p_id = 1;
        state.settings.current_player_turn_id = p_id;

        let mut tribe = TribeState::default();
        tribe.id = p_id;
        tribe.tribe_type = TribeType::Luxidoor;
        tribe.starting_tile_coords = Coords::from_xy(0, 0, 11);

        let mut city = CityState::default();
        city.tile_index = 0; // matching (0,0) for size 11
        city.level = 3;
        city.owner = p_id;
        tribe.cities.push(city);

        state.tribes.insert(p_id, tribe);

        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        generate_reward_moves(&state, &mut moves);

        assert_eq!(
            moves.len(),
            0,
            "Luxidoor level 3 city should have no pending rewards"
        );
    }

    #[test]
    fn test_luxidoor_reward_at_level_4() {
        let mut state = GameState::default();
        let p_id = 1;
        state.settings.current_player_turn_id = p_id;

        let mut tribe = TribeState::default();
        tribe.id = p_id;
        tribe.tribe_type = TribeType::Luxidoor;
        tribe.starting_tile_coords = Coords::from_xy(0, 0, 11);

        let mut city = CityState::default();
        city.tile_index = 0;
        city.level = 4;
        city.owner = p_id;
        tribe.cities.push(city);

        state.tribes.insert(p_id, tribe);

        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        generate_reward_moves(&state, &mut moves);

        // Should have 2 moves (opt1, opt2) for the reward slot
        assert_eq!(
            moves.len(),
            2,
            "Luxidoor level 4 city should have 2 reward options"
        );

        // Slot should be 3 (PopGrowth/BorderGrowth)
        // We can check the description or the move structure
        let desc = moves[0].describe(&state);
        assert!(
            desc.contains("PopGrowth") || desc.contains("BorderGrowth"),
            "Should be slot 3 rewards"
        );
    }

    #[test]
    fn test_normal_tribe_rewards() {
        let mut state = GameState::default();
        let p_id = 1;
        state.settings.current_player_turn_id = p_id;

        let mut tribe = TribeState::default();
        tribe.id = p_id;
        tribe.tribe_type = TribeType::Imperius;
        tribe.starting_tile_coords = Coords::from_xy(0, 0, 11);

        let mut city = CityState::default();
        city.tile_index = 0;
        city.level = 2;
        city.owner = p_id;
        tribe.cities.push(city);

        state.tribes.insert(p_id, tribe);

        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        generate_reward_moves(&state, &mut moves);

        assert_eq!(
            moves.len(),
            2,
            "Normal tribe level 2 city should have 2 reward options"
        );
        let desc = moves[0].describe(&state);
        assert!(
            desc.contains("Explorer") || desc.contains("Workshop"),
            "Should be slot 1 rewards"
        );
    }
}
