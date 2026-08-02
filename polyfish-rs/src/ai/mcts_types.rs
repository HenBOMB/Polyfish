#[derive(Debug)]
pub struct MoveVisit {
    pub move_type: crate::types::MoveType,
    pub visits: f32,
    pub source_idx: Option<usize>,
    pub target_idx: Option<usize>,
    pub structure_type: Option<crate::types::StructureType>,
    pub unit_type: Option<crate::types::UnitType>,
    pub tech_type: Option<crate::types::TechnologyType>,
    pub ability_type: Option<crate::types::AbilityType>,
    pub reward_type: Option<crate::types::CityRewardType>,
}
