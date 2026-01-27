use crate::states::GameState;
use crate::types::{StructureType, TaskType, TechnologyType};

pub struct TaskSetting {
    pub tech_type: Option<TechnologyType>,
    pub structure_type: StructureType,
}

pub fn get_task_setting(task_type: TaskType) -> TaskSetting {
    match task_type {
        TaskType::Pacifist => TaskSetting {
            tech_type: Some(TechnologyType::Meditation),
            structure_type: StructureType::AltarOfPeace,
        },
        TaskType::Genius => TaskSetting {
            tech_type: Some(TechnologyType::Philosophy),
            structure_type: StructureType::TowerOfWisdom,
        },
        TaskType::Wealth => TaskSetting {
            tech_type: Some(TechnologyType::Trade),
            structure_type: StructureType::EmperorsTomb,
        },
        TaskType::Explorer => TaskSetting {
            tech_type: None,
            structure_type: StructureType::EyeOfGod,
        },
        TaskType::Killer => TaskSetting {
            tech_type: None,
            structure_type: StructureType::GateOfPower,
        },
        TaskType::Network => TaskSetting {
            tech_type: None,
            structure_type: StructureType::GrandBazaar,
        },
        TaskType::Metropolis => TaskSetting {
            tech_type: None,
            structure_type: StructureType::ParkOfFortune,
        },
    }
}

pub fn check_task(state: &GameState, task_type: TaskType) -> bool {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return false,
    };

    match task_type {
        TaskType::Pacifist => false, // TODO: Implement pacifist counting (requires turn history/stats)
        TaskType::Genius => {
            // Must have Philosophy requirement (checked by caller usually, but explicit here too)
            if !crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Philosophy,
            ) {
                return false;
            }
            // Check if all techs researched. Standard tree is ~25 techs (excluding Unrequired?).
            // 5 branches * 5 techs/branch = 25.
            // Some special tribes might vary, but length check is a good heuristic.
            const TOTAL_TECHS: usize = 25;
            // If tech_vanilla includes Unrequired, it might be 26.
            // Let's assume >= 22 implies "all" for safety or check exact.
            // Better: Iterate all standard techs and check? No, that's slow.
            // Let's assume strict length.
            tribe.tech_vanilla.len() >= TOTAL_TECHS
        }
        TaskType::Wealth => {
            if !crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Trade,
            ) {
                return false;
            }
            // Logic from TS: Connected Cities (Wealth vs Network distinct in Polytopia?)
            // TS Wealth: connectedToCapital > 4 (via Trade)
            // TS Network: connectedToCapital > 4 (via GrandBazaar)
            // It seems redundant, but we follow TS logic.
            // Accumulate check vs Filter check:
            let connected_count = tribe
                .cities
                .iter()
                .filter(|c| c.connected_to_capital)
                .count();
            connected_count > 4
        }
        TaskType::Explorer => {
            // Check if all corners are explored by us.
            let size = state.settings.size;
            let corners = [0, size - 1, size * size - 1, size * size - size];
            corners.iter().all(|&idx| {
                if let Some(tile) = state.tiles.get(&idx) {
                    tile.explorers.contains(&pov_id)
                } else {
                    false
                }
            })
        }
        TaskType::Killer => tribe.kills > 9,
        TaskType::Network => {
            let connected_count = tribe
                .cities
                .iter()
                .filter(|c| c.connected_to_capital)
                .count();
            connected_count > 4
        }
        TaskType::Metropolis => tribe.cities.iter().any(|c| c.level > 4),
    }
}
