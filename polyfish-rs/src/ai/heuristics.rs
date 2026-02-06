use crate::functions::{get_defense_bonus, get_tribe_spt, get_unit_max_health, has_effect};
use crate::states::{GameState, PlayerId, UnitState};
use crate::types::{EffectType, UnitType};
use std::collections::HashMap;

/// Scores for each unit type based on their base strength in the meta
pub struct UnitValues {
    values: HashMap<UnitType, f32>,
}

impl UnitValues {
    pub fn new() -> Self {
        let mut values = HashMap::new();

        // Super Units (S tier)
        values.insert(UnitType::Shaman, 1.00);
        values.insert(UnitType::FireDragon, 0.95);
        values.insert(UnitType::BabyDragon, 0.90);
        values.insert(UnitType::DragonEgg, 0.85);
        values.insert(UnitType::Centipede, 0.83);
        values.insert(UnitType::Segment, 0.81);
        values.insert(UnitType::Crab, 0.80);
        values.insert(UnitType::Giant, 0.74);
        values.insert(UnitType::Gaami, 0.70);
        values.insert(UnitType::Juggernaut, 0.50);

        // Spawnable Units (S/A tier)
        values.insert(UnitType::Rider, 0.60);
        values.insert(UnitType::Hexapod, 0.60);
        values.insert(UnitType::BattleSled, 0.50);
        values.insert(UnitType::Amphibian, 0.60);
        values.insert(UnitType::Archer, 0.47);
        values.insert(UnitType::Knight, 0.46);
        values.insert(UnitType::Cloak, 0.45);

        // Naval
        values.insert(UnitType::Dinghy, 0.43);
        values.insert(UnitType::Dagger, 0.43);
        values.insert(UnitType::Pirate, 0.43);
        values.insert(UnitType::Rammer, 0.41);

        // Standard
        values.insert(UnitType::Scout, 0.40);
        values.insert(UnitType::Warrior, 0.39);
        values.insert(UnitType::Defender, 0.38);
        values.insert(UnitType::Catapult, 0.37);
        values.insert(UnitType::Swordsman, 0.29);
        values.insert(UnitType::MindBender, 0.15);

        // Others default to low score
        Self { values }
    }

    pub fn get(&self, unit_type: UnitType) -> f32 {
        *self.values.get(&unit_type).unwrap_or(&0.1)
    }
}

// Global instance for lookup
lazy_static::lazy_static! {
    pub static ref UNIT_VALUES: UnitValues = UnitValues::new();
}

/// Calculate a unit's power score based on health, status, and position
pub fn assess_unit_power(game: &GameState, unit: &UnitState) -> f32 {
    let mut score = UNIT_VALUES.get(unit.unit_type);

    // Health Multiplier (Linear)
    let max_hp = get_unit_max_health(unit) as f32;
    let hp_mult = unit.health as f32 / max_hp;
    score *= hp_mult;

    // Defense Bonus (Walls/Terrain)
    // Defense bonus is 1.0 (none), 1.5 (terrain), 4.0 (walls)
    // We normalize this slightly to not overvalue passive defense
    let defense = get_defense_bonus(game, unit);
    let defense_mult = 1.0 + (defense - 1.0) * 0.2; // dampen the effect
    score *= defense_mult;

    // Status Effects
    if has_effect(unit, EffectType::Poison) {
        score *= 0.7;
    }
    if has_effect(unit, EffectType::Boost) {
        score *= 1.2;
    }
    if has_effect(unit, EffectType::Frozen) {
        score *= 0.1; // Useless for a turn
    }

    // Veteran bonus (more max HP + heal)
    if unit.veteran {
        score *= 1.1;
    }

    // Kills (experience)
    score += (unit.kills as f32) * 0.05;

    score
}

/// Calculate heuristic scores for the game state (Economy, Military)
/// Returns (EcoScore, MilScore) normalized roughly 0.0 to 1.0
pub fn evaluate_state_heuristics(state: &GameState, player_id: PlayerId) -> (f32, f32) {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return (0.0, 0.0);
    }
    let tribe = tribe_opt.unwrap();

    // --- Economy Score ---
    // SPT / 30 (Soft cap at 30 SPT for 1.0 score)
    let spt = get_tribe_spt(state, tribe) as f32;
    let eco_score = (spt / 30.0).clamp(0.0, 1.0);

    // --- Military Score ---
    // Sum of unit power / 20 (Soft cap at ~20 strong units)
    let mut mil_sum = 0.0;
    for unit in &tribe.units {
        // We only assess our own units here
        // Note: Ideally we'd pass 'game' but we only have 'state' here. output is valid enough.
        mil_sum += assess_unit_power(state, unit);
    }
    // Average unit power is ~0.4. 20 units * 0.4 = 8.0.
    // Let's normalize so a strong army gives 1.0
    let mil_score = (mil_sum / 10.0).clamp(0.0, 1.0);

    (eco_score, mil_score)
}
