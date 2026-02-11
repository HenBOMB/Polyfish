use crate::functions::{get_defense_bonus, get_unit_max_health, has_effect};
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

        // Direct meta-scoring: 0.0-1.0 based on actual unit strength in combat.
        // Stats (HP/buffs/defense) are already handled by assess_unit_power(),
        // so base scores reflect strategic value, mobility, skills, and synergy.

        // === Super Units (S-tier: 0.70-0.95) ===
        values.insert(UnitType::FireDragon, 0.95); // Fly, Splash, Range 2, 20HP — best unit
        values.insert(UnitType::BabyDragon, 0.80); // Fly, Dash, Escape, grows into FireDragon
        values.insert(UnitType::Crab, 0.85); // 40HP, Escape, AutoFlood, Amphibious
        values.insert(UnitType::Giant, 0.80); // 40HP, raw power
        values.insert(UnitType::Juggernaut, 0.78); // 40HP water Giant, Stomp
        values.insert(UnitType::Centipede, 0.75); // 20HP, Eat spawns Segments
        values.insert(UnitType::LivingIsland, 0.70); // 20HP water, Stomp+Poison
        values.insert(UnitType::Shaman, 0.70); // Convert + Swarm, huge strategic value

        // === Strong Units (A-tier: 0.50-0.69) ===
        values.insert(UnitType::Knight, 0.65); // Persist (chain kills), 3 movement
        values.insert(UnitType::Tridention, 0.65); // Dash+Persist, Range 2, Amphibious
        values.insert(UnitType::Rider, 0.60); // Dash+Escape, 2 movement — versatile
        values.insert(UnitType::Doomux, 0.58); // 3 movement, Explode, 20HP
        values.insert(UnitType::Amphibian, 0.55); // Rider but amphibious
        values.insert(UnitType::Hexapod, 0.55); // Dash+Escape+Sneak, glass cannon
        values.insert(UnitType::Rammer, 0.55); // 3 movement naval, Dash+Carry
        values.insert(UnitType::Bomber, 0.55); // Range 3 Splash on water
        values.insert(UnitType::Mantis, 0.55); // Cymanti super, 20HP
        values.insert(UnitType::Catapult, 0.52); // Range 3, fragile but deadly
        values.insert(UnitType::Archer, 0.50); // Range 2, cheap, Dash
        values.insert(UnitType::Scout, 0.50); // Range 2, 3 movement naval
        values.insert(UnitType::Exida, 0.50); // Range 3, Splash+Poison

        // === Mid Units (B-tier: 0.30-0.49) ===
        values.insert(UnitType::Swordsman, 0.48); // Tanky (15HP, 3def) but slow
        values.insert(UnitType::Cloak, 0.47); // Infiltrate, Hide, scout utility
        values.insert(UnitType::Boomchi, 0.45); // Explode+Amphibious, niche
        values.insert(UnitType::Moth, 0.45); // Fly+Infiltrate, evolved Larva
        values.insert(UnitType::Defender, 0.43); // High def but low attack
        values.insert(UnitType::Pirate, 0.42); // Surprise+Carry, naval agent
        values.insert(UnitType::Dagger, 0.40); // Surprise, independent
        values.insert(UnitType::Polytaur, 0.40); // Cheap ranged independent
        values.insert(UnitType::Phychi, 0.40); // Fly+DoubleAttack, fragile
        values.insert(UnitType::Raychi, 0.40); // Water-only, fast
        values.insert(UnitType::Warrior, 0.38); // Basic unit
        values.insert(UnitType::Segment, 0.35); // Independent, Explode option
        values.insert(UnitType::Kiton, 0.35); // Cheap Poison defender
        values.insert(UnitType::Dinghy, 0.33); // Naval agent, Hide+Infiltrate
        values.insert(UnitType::MindBender, 0.30); // Convert niche, no attack

        // === Weak / Transitional (C-tier: 0.05-0.29) ===
        values.insert(UnitType::DragonEgg, 0.25); // No attack, grows into BabyDragon
        values.insert(UnitType::Larva, 0.20); // Transition, grows into Moth
        values.insert(UnitType::Raft, 0.15); // No attack, just a boat
        values.insert(UnitType::InsectEgg, 0.10); // Immobile, grows into Larva

        // === Disabled (Polaris — cost=-1 in settings) ===
        values.insert(UnitType::Gaami, 0.05);
        values.insert(UnitType::Mooni, 0.05);
        values.insert(UnitType::BattleSled, 0.05);
        values.insert(UnitType::IceArcher, 0.05);
        values.insert(UnitType::IceFortress, 0.05);

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
/// Returns Unit power 0.0 to 1.0 associated with a confidence of the unit's strength
pub fn assess_unit_power(game: &GameState, unit: &UnitState) -> f32 {
    // 1. Base Power (40%) - derived from meta strength
    let base_score = UNIT_VALUES.get(unit.unit_type);

    // 2. Health (30%) - % of max HP
    let max_hp = get_unit_max_health(unit) as f32;
    let hp_score = (unit.health as f32 / max_hp).clamp(0.0, 1.0);

    // 3. Status (20%) - Buffs/Debuffs
    // Start at 0.5 (neutral)
    let mut status_val = 0.5;

    if unit.veteran {
        status_val += 0.2;
    }
    if has_effect(unit, EffectType::Boost) {
        status_val += 0.15;
    }

    // Kills (max 3 -> +0.15)
    status_val += unit.kills.min(3) as f32 * 0.05;

    // Debuffs
    if has_effect(unit, EffectType::Poison) {
        status_val -= 0.2;
    }
    if has_effect(unit, EffectType::Frozen) {
        status_val -= 0.4;
    } // Big penalty

    let status_score = status_val.clamp(0.0, 1.0);

    // 4. Defense (10%) - Terrain/Walls
    // Defense bonus ranges from 1.0 (none) to 4.0 (walled city)
    // We map 1.0 -> 0.0 and 4.0 -> 1.0
    let def_bonus = get_defense_bonus(game, unit);
    let def_score = ((def_bonus - 1.0) / 3.0).clamp(0.0, 1.0);

    // 5. Loneliness / Support (Penalty)
    // If unit is weak/mid (< 0.5 base) and has no friends nearby, penalize.
    let mut loneliness_penalty = 0.0;
    if base_score < 0.6 {
        // Check 2-tile radius for friends
        let adj = crate::functions::get_adjacent_indices(game, unit.coords.idx, 2);
        let friends = adj
            .iter()
            .filter(|&&idx| {
                if let Some(other) = game
                    .tribes
                    .get(&unit.owner)
                    .and_then(|t| t.units.iter().find(|u| u.coords.idx == idx))
                {
                    other.coords.idx != unit.coords.idx // Don't count self
                } else {
                    false
                }
            })
            .count();

        if friends == 0 {
            loneliness_penalty = 0.15;
        } else if friends == 1 {
            loneliness_penalty = 0.05;
        }
    }

    // Final Weighted Sum
    // Weights: Base=0.4, Health=0.3, Status=0.2, Defense=0.1
    // Loneliness is a flat penalty
    let final_score =
        (base_score * 0.4) + (hp_score * 0.3) + (status_score * 0.2) + (def_score * 0.1)
            - loneliness_penalty;

    final_score.clamp(0.0, 1.0)
}

// Evaluates the power of the army, returns a score between 0.0 and 1.0
pub fn evaluate_army(state: &GameState, player_id: PlayerId) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    let tribe = tribe_opt.unwrap();

    // --- 2. Military Score (0.0 - 1.0) ---
    // Sum of unit power.
    // A full army of 20 strong units (avg 0.7) = 14.0 score.
    // Soft cap at 20.0 (allows for huge armies to saturate, but typically < 1.0)
    let mut score_army = 0.0;
    for unit in &tribe.units {
        score_army += assess_unit_power(state, unit);
    }

    let progress = (state.settings.turn as f32 / state.settings.max_turns as f32).clamp(0.0, 1.0);
    let mut max_units;

    if progress < 0.3 {
        // Early Game: At least 2.0 units per city + 1 extra unit
        max_units = tribe.cities.len() as f32 * 2.0 + 1.0;
    } else if progress < 0.7 {
        // Mid Game: At least 2.0 units per city + 2 extra units
        max_units = tribe.cities.len() as f32 * 2.0 + 2.0;
    } else {
        // Late Game: At least 4.0 units per city + 4 extra units
        max_units = tribe.cities.len() as f32 * 4.0 + 4.0;
    };

    max_units = max_units.min(crate::states::default_max_units() as f32);

    (score_army / max_units).clamp(0.0, 1.0)
}
