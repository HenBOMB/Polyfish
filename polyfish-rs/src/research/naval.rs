use crate::functions::{
    get_adjacent_indices, get_chebyshev_distance, get_structure_type_at, get_unit_at,
};
use crate::moves::Move;
use crate::settings::structures::get_structure_setting;
use crate::settings::technology::has_technology;
use crate::states::{GameState, PlayerId};
use crate::types::{MoveType, StructureType, TechnologyType, TerrainType, UnitType};

pub(super) const NAVAL_PRIORITY: u8 = 7;

/// Scores only explored terrain, known objectives, and the acting player's ships and economy.
pub(super) struct NavalPolicy<'a> {
    state: &'a GameState,
    pov: PlayerId,
    ports: Vec<i32>,
    land_goals: Vec<i32>,
    has_harbor: bool,
    saving: bool,
}

impl<'a> NavalPolicy<'a> {
    pub(super) fn new(state: &'a GameState) -> Self {
        let pov = state.settings.current_player_turn_id;
        let navigable_shores: Vec<_> = state
            .tiles
            .values()
            .filter(|tile| {
                tile.explorers.contains(&pov)
                    && tile.owner == pov
                    && tile.terrain_type == TerrainType::Water
                    && get_adjacent_indices(state, tile.coords.idx, 1)
                        .into_iter()
                        .any(|idx| {
                            let neighbor = &state.tiles[&idx];
                            !neighbor.explorers.contains(&pov) || neighbor.is_water_terrain()
                        })
            })
            .collect();
        let ports: Vec<_> = navigable_shores
            .iter()
            .filter_map(|tile| {
                let idx = tile.coords.idx;
                (get_structure_type_at(state, idx) == Some(StructureType::Port)).then_some(idx)
            })
            .collect();
        let has_harbor = navigable_shores
            .iter()
            .any(|tile| get_structure_type_at(state, tile.coords.idx).is_none());
        let tribe = &state.tribes[&pov];
        let port_cost = get_structure_setting(StructureType::Port)
            .cost
            .unwrap_or_default();
        let saving = has_harbor
            && ports.is_empty()
            && tribe.stars < port_cost
            && has_technology(&tribe.tech_vanilla, TechnologyType::Fishing);
        let land_goals = super::visible_goals(state)
            .into_iter()
            .filter(|idx| !state.tiles[idx].is_water_terrain())
            .collect();
        Self {
            state,
            pov,
            ports,
            land_goals,
            has_harbor,
            saving,
        }
    }

    pub(super) fn save_for_port(&self, action: &dyn Move) -> bool {
        self.saving
            && matches!(
                action.move_type(),
                MoveType::Build | MoveType::Harvest | MoveType::Research | MoveType::Summon
            )
    }

    fn distance(&self, from: i32, to: i32) -> i32 {
        get_chebyshev_distance(from, to, self.state.settings.size)
    }

    pub(super) fn priority(&self, action: &dyn Move) -> Option<u8> {
        match action.move_type() {
            MoveType::Capture => Some(11),
            MoveType::Research => match action.tech_type().ok()? {
                TechnologyType::Fishing if self.has_harbor && self.ports.is_empty() => Some(9),
                TechnologyType::Sailing if !self.ports.is_empty() => Some(9),
                _ => None,
            },
            MoveType::Build if action.structure_type().ok()? == StructureType::Port => {
                let idx = action.target_idx().ok()? as i32;
                if !self.ports.is_empty() {
                    return Some(2);
                }
                let coast = get_adjacent_indices(self.state, idx, 1)
                    .into_iter()
                    .any(|idx| {
                        let tile = &self.state.tiles[&idx];
                        !tile.explorers.contains(&self.pov) || tile.is_water_terrain()
                    });
                Some(if coast { 10 } else { 2 })
            }
            MoveType::Summon
                if action.source_idx().ok().is_some_and(|src| {
                    get_unit_at(self.state, src as i32)
                        .is_some_and(|unit| unit.unit_type == UnitType::Raft)
                }) =>
            {
                Some(if action.unit_type().ok()? == UnitType::Scoutship {
                    9
                } else {
                    6
                })
            }
            MoveType::Step => self.step_priority(
                action.source_idx().ok()? as i32,
                action.target_idx().ok()? as i32,
            ),
            _ => None,
        }
    }

    fn step_priority(&self, from: i32, to: i32) -> Option<u8> {
        let unit = get_unit_at(self.state, from)?;
        let nearest_port = |idx| {
            self.ports
                .iter()
                .map(|&port| self.distance(idx, port))
                .min()
        };
        let nearest_goal = |idx| {
            self.land_goals
                .iter()
                .map(|&goal| self.distance(idx, goal))
                .min()
        };
        if !unit.unit_type.is_naval() {
            if self.ports.contains(&to) {
                return Some(8);
            }
            if nearest_port(to) < nearest_port(from)
                && self.state.tribes[&self.pov]
                    .units
                    .iter()
                    .all(|unit| !unit.unit_type.is_naval())
            {
                return Some(5);
            }
            return None;
        }

        let target = &self.state.tiles[&to];
        if !target.is_water_terrain() {
            if self.land_goals.contains(&to) {
                return Some(10);
            }
            if target.owner == self.pov || nearest_port(to) == Some(1) {
                return Some(0);
            }
            if nearest_goal(to) < nearest_goal(from) {
                return Some(8);
            }
        }
        if to == unit.prev_coords.idx {
            return Some(0);
        }
        let explores = get_adjacent_indices(self.state, to, 1)
            .into_iter()
            .filter(|idx| !self.state.tiles[idx].explorers.contains(&self.pov))
            .count();
        if target.is_water_terrain() {
            if nearest_goal(to) < nearest_goal(from)
                && nearest_goal(to).is_some_and(|distance| distance != 0)
            {
                return Some(8);
            }
            return Some(if explores > 0 {
                7
            } else if nearest_port(to) > nearest_port(from) {
                4
            } else {
                1
            });
        }
        Some(if explores > 0 { 6 } else { 2 })
    }
}
