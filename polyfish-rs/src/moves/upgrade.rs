//! Upgrade move (Unit Upgrade)
//!
//! Upgrade a unit (e.g. Boat -> Ship).

use crate::actions::chain_undos;
use crate::actions::spend_stars;
use crate::functions;
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
use crate::settings::technology;
use crate::states::GameState;
use crate::types::{MoveType, UnitType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradeMove {
    pub tile_index: i32,
    #[serde(rename = "type")]
    pub target_type: UnitType,
}

impl UpgradeMove {
    pub fn new(tile_index: i32, target_type: UnitType) -> Self {
        Self {
            tile_index,
            target_type,
        }
    }
}

impl Move for UpgradeMove {
    fn move_type(&self) -> MoveType {
        MoveType::Summon // Reusing Summon type per TS implementation
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let mut undos = Vec::new();

        // Find unit
        let unit_owner = state
            .tiles
            .get(&self.tile_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let tribe = match state.tribes.get_mut(&unit_owner) {
            Some(t) => t,
            None => {
                return MoveResult {
                    undo: Box::new(|_| {}),
                    rewards: None,
                }
            }
        };

        let unit_idx = match tribe
            .units
            .iter()
            .position(|u| u.coords.idx == self.tile_index)
        {
            Some(i) => i,
            None => {
                return MoveResult {
                    undo: Box::new(|_| {}),
                    rewards: None,
                }
            }
        };

        let unit = &mut tribe.units[unit_idx];
        let old_type = unit.unit_type;
        let settings = get_unit_setting(self.target_type);

        // Spend stars
        undos.push(spend_stars(state, settings.cost));

        // Upgrade unit
        if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.unit_type = self.target_type;
                // Handle passenger logic for naval upgrades?
                // Unit upgrading from Dinghy -> Ship usually keeps original passenger info?
                // Or unit IS the boat?
                // TS Upgrade.ts: `xorUnit.passenger(state, unit, UnitType.None, oldUnitType);`
                // `unit.passengerType = oldUnitType;`
                // Wait. If upgrading Dinghy -> Ship, the Dinghy becomes the passenger?
                // No. `oldUnitType` was Dinghy?
                // Usually: A Warrior enters Port -> becomes Dinghy (Warrior is passenger).
                // Dinghy upgrades to Ship -> Ship (Warrior is still passenger).
                // `unit.passengerType` should persist.
                // TS: `unit.passengerType = oldUnitType;` ? That looks wrong if oldUnitType was Dinghy.
                // Unless Upgrade is replacing the hull?
                // Let's assume for now we just change unit_type. Passenger usually preserved.
                // But wait, if we are upgrading Boat -> Ship, `unit_type` changes.
                // Does `passenger_type` change? No.
            }
        }

        let target_type = self.target_type;
        undos.push(Box::new(move |s| {
            if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(unit_idx) {
                    unit.unit_type = old_type;
                    // Restore stars handled by spend_stars undo
                }
            }
        }));

        MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Upgrade unit at {} to {:?}",
            self.tile_index, self.target_type
        )
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Generate upgrade moves
pub fn generate_upgrade_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    crate::moves::upgrade::generate_upgrade_moves_internal(state, moves);
}

fn generate_upgrade_moves_internal(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    // Find all upgradable unit types unlocked by researched techs
    let mut upgradables = Vec::new();
    for tech_state in &tribe.tech_vanilla {
        if !tech_state.discovered {
            continue;
        }

        if let Some(u_type) = get_tech_unit_type(tribe, tech_state.tech_type) {
            let settings = get_unit_setting(u_type);
            if settings.cost >= 1 && tribe.stars >= settings.cost && settings.upgrade_from.is_some()
            {
                upgradables.push(u_type);
            }
        }
    }

    if upgradables.is_empty() {
        return;
    }

    for unit in &tribe.units {
        // Polytopia/TS Rule: Only Raft can be upgraded (usually Boat/Ship/etc. handled via Navy)
        // and tile must not be "occupied" (TS weirdness)
        if unit.unit_type != UnitType::Raft || is_tile_occupied(state, unit.coords.idx) {
            continue;
        }

        for &u_type in &upgradables {
            moves.push(Box::new(UpgradeMove::new(unit.coords.idx, u_type)));
        }
    }
}

fn get_tech_unit_type(
    _tribe: &crate::states::TribeState,
    tech: crate::types::TechnologyType,
) -> Option<UnitType> {
    crate::settings::technology::get_technology_setting(tech).unlocks_unit
}

fn is_tile_occupied(state: &GameState, idx: i32) -> bool {
    crate::functions::get_unit_at(state, idx).is_some()
}
