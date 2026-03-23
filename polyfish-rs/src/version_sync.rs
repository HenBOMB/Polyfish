use crate::states::GameState;

pub fn get_burn_forest_cost(gs: &GameState) -> i32 {
    if gs.settings.version <= 104 { 4 } else { 3 }
}

pub fn get_clear_forest_stars(gs: &GameState) -> i32 {
    if gs.settings.version <= 104 { 2 } else { 1 }
}

pub fn get_polytaur_cost(gs: &GameState) -> i32 {
    if gs.settings.version <= 104 { 2 } else { 1 }
}
