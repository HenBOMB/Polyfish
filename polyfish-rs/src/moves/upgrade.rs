//! Upgrade move (Unit Upgrade)
//!
//! Upgrade a unit (e.g. Boat -> Ship).

use crate::functions::{get_unit_at, is_at_peace};
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
use crate::states::{GameState, PlayerId};
use crate::types::{MoveType, UnitType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpgradeMove {
    pub src_index: i32,
    #[serde(rename = "type")]
    pub target_type: UnitType,
}

impl UpgradeMove {
    pub fn new(src_index: i32, target_type: UnitType) -> Self {
        Self {
            src_index,
            target_type,
        }
    }
}

impl Move for UpgradeMove {
    fn move_type(&self) -> MoveType {
        MoveType::Summon // Reusing Summon type per TS implementation
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;

        let unit = get_unit_at(state, self.src_index).ok_or("No unit to upgrade")?;
        let settings = get_unit_setting(self.target_type);
        if unit.owner != pov_id
            || settings.upgrade_from != Some(unit.unit_type)
            || !can_upgrade_at(state, self.src_index, pov_id)
        {
            return Err("Unit cannot upgrade here".into());
        }
        if let Some(tribe) = state.tribes.get(&pov_id) {
            if !tribe.tech_vanilla.iter().any(|tech| {
                crate::settings::technology::get_unlocked_units(tech.tech_type, tribe.tribe_type)
                    .contains(&self.target_type)
            }) {
                return Err("Upgrade requires an unlocked unit".into());
            }
            if tribe.stars < settings.cost {
                return Err(format!(
                    "Insufficient stars for upgrade: need {}, have {}",
                    settings.cost, tribe.stars
                ));
            }
        } else {
            return Err("Tribe not found".to_string());
        }

        // Delegate to action
        let undo = crate::actions::units::upgrade_unit(state, self.src_index, self.target_type)?;

        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Upgrade unit at {} to {:?}",
            self.src_index, self.target_type
        )
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "src": self.src_index,
            "type": self.target_type,
            "upgrade": true,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn unit_type(&self) -> Result<UnitType, String> {
        Ok(self.target_type)
    }
}

fn can_upgrade_at(state: &GameState, idx: i32, owner: PlayerId) -> bool {
    state.tiles.get(&idx).is_some_and(|tile| {
        tile.owner == owner || (tile.owner > 0 && is_at_peace(state, owner, tile.owner))
    })
}

/// Generate upgrade moves
pub fn generate_upgrade_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    // Find all upgradable unit types unlocked by researched techs
    let mut upgradables = Vec::new();
    for tech_state in &tribe.tech_vanilla {
        // We use all researched techs for upgrades, even if not "discovered" (simulation)
        for u_type in
            crate::settings::technology::get_unlocked_units(tech_state.tech_type, tribe.tribe_type)
        {
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
        if unit.unit_type != UnitType::Raft || !can_upgrade_at(state, unit.coords.idx, pov_id) {
            continue;
        }

        for &u_type in &upgradables {
            moves.push(Box::new(UpgradeMove::new(unit.coords.idx, u_type)));
        }
    }
}
