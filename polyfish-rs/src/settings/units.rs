//! Unit settings and stats

use crate::types::{SkillType, TribeType, UnitType};
use std::collections::HashSet;

/// Unit stat configuration
#[derive(Debug, Clone)]
pub struct UnitSetting {
    pub cost: i32,
    pub attack: f32,
    pub movement: i32,
    pub defense: f32,
    pub range: i32,
    pub health: f32, // todo move to Option to enfoce passenger inheritance?
    pub upgrade_from: Option<UnitType>,
    pub is_unique: bool,
    pub skills: HashSet<SkillType>,
    pub is_super: bool,
    pub tribe_type: TribeType,
}

impl Default for UnitSetting {
    fn default() -> Self {
        Self {
            cost: 0,
            attack: 0.0,
            movement: 1,
            defense: 0.0,
            range: 1,
            health: 10.0,
            upgrade_from: None,
            is_unique: false,
            skills: HashSet::new(),
            is_super: false,
            tribe_type: TribeType::None,
        }
    }
}

/// Helper macro to create skill sets
macro_rules! skills {
    ($($skill:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(set.insert($skill);)*
        set
    }};
}

/// Get unit settings by type
pub fn get_unit_setting(unit_type: UnitType) -> UnitSetting {
    use SkillType as S;
    use UnitType as U;

    match unit_type {
        U::None => UnitSetting {
            cost: -1,
            attack: -1.0,
            health: 1.0,
            ..Default::default()
        },

        U::Warrior => UnitSetting {
            cost: 2,
            attack: 2.0,
            movement: 1,
            defense: 2.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Dash, S::Fortify],
            ..Default::default()
        },
        U::Rider => UnitSetting {
            cost: 3,
            attack: 2.0,
            movement: 2,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Dash, S::Escape, S::Fortify],
            ..Default::default()
        },
        U::Knight => UnitSetting {
            cost: 8,
            attack: 3.5,
            movement: 3,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Dash, S::Persist, S::Fortify],
            ..Default::default()
        },
        U::Defender => UnitSetting {
            cost: 3,
            attack: 1.0,
            movement: 1,
            defense: 3.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Fortify],
            ..Default::default()
        },
        U::Catapult => UnitSetting {
            cost: 8,
            attack: 4.0,
            movement: 1,
            defense: 0.0,
            range: 3,
            health: 10.0,
            skills: skills![S::Stiff],
            ..Default::default()
        },
        U::Archer => UnitSetting {
            cost: 3,
            attack: 2.0,
            movement: 1,
            defense: 1.0,
            range: 2,
            health: 10.0,
            skills: skills![S::Dash, S::Fortify],
            ..Default::default()
        },
        U::MindBender => UnitSetting {
            cost: 5,
            attack: 0.0,
            movement: 1,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::HealOthers, S::Convert, S::Stiff],
            ..Default::default()
        },
        U::Swordsman => UnitSetting {
            cost: 5,
            attack: 3.0,
            movement: 1,
            defense: 3.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Dash],
            ..Default::default()
        },
        U::Giant => UnitSetting {
            cost: 10,
            attack: 5.0,
            movement: 1,
            defense: 4.0,
            range: 1,
            health: 40.0,
            is_super: true,
            skills: skills![S::Static],
            ..Default::default()
        },

        // Agent units
        U::Cloak => UnitSetting {
            cost: 8,
            attack: 0.0,
            movement: 2,
            defense: 0.5,
            range: 1,
            health: 5.0,
            skills: skills![
                S::Hide,
                S::Infiltrate,
                S::Dash,
                S::Scout,
                S::Creep,
                S::Static
            ],
            ..Default::default()
        },
        U::Dagger => UnitSetting {
            cost: 2,
            attack: 2.0,
            movement: 1,
            defense: 2.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Dash, S::Surprise, S::Independent, S::Static],
            ..Default::default()
        },
        U::Pirate => UnitSetting {
            cost: 2,
            attack: 2.0,
            movement: 2,
            defense: 2.0,
            range: 1,
            health: 10.0,
            skills: skills![
                S::Carry,
                S::Float,
                S::Dash,
                S::Surprise,
                S::Independent,
                S::Static
            ],
            ..Default::default()
        },
        U::Dinghy => UnitSetting {
            cost: 2,
            attack: 0.0,
            movement: 2,
            defense: 0.5,
            range: 1,
            health: 5.0,
            skills: skills![
                S::Carry,
                S::Float,
                S::Hide,
                S::Infiltrate,
                S::Dash,
                S::Scout,
                S::Static
            ],
            ..Default::default()
        },

        // Navy
        U::Raft => UnitSetting {
            cost: 0,
            attack: 0.0,
            movement: 2,
            defense: 1.0,
            range: 2,
            health: 10.0,
            skills: skills![S::Carry, S::Float, S::Static, S::Stiff],
            ..Default::default()
        },
        U::Scoutship => UnitSetting {
            cost: 5,
            attack: 2.0,
            movement: 3,
            defense: 1.0,
            range: 2,
            health: 10.0, // inherits
            upgrade_from: Some(U::Raft),
            skills: skills![S::Carry, S::Dash, S::Float, S::Static, S::Scout],
            ..Default::default()
        },
        U::Rammership => UnitSetting {
            cost: 5,
            attack: 3.0,
            movement: 3,
            defense: 3.0,
            range: 1,
            health: 10.0,
            upgrade_from: Some(U::Raft),
            skills: skills![S::Dash, S::Float, S::Carry, S::Static],
            ..Default::default()
        },
        U::Bomber => UnitSetting {
            cost: 15,
            attack: 3.0,
            movement: 2,
            defense: 2.0,
            range: 3,
            health: 10.0,
            upgrade_from: Some(U::Raft),
            skills: skills![S::Carry, S::Float, S::Splash, S::Stiff, S::Static],
            ..Default::default()
        },
        U::Juggernaut => UnitSetting {
            cost: 10,
            attack: 4.0,
            movement: 2,
            defense: 4.0,
            range: 1,
            health: 40.0,
            is_super: true,
            skills: skills![S::Float, S::Carry, S::Stiff, S::Stomp, S::Static],
            ..Default::default()
        },

        // Aquarion
        U::Crab => UnitSetting {
            cost: 10,
            attack: 4.0,
            movement: 2,
            defense: 5.0,
            range: 1,
            health: 40.0,
            is_super: true,
            skills: skills![S::Escape, S::Static, S::AutoFlood, S::Amphibious],
            tribe_type: TribeType::Aquarion,
            ..Default::default()
        },
        U::Amphibian => UnitSetting {
            cost: 3,
            attack: 2.0,
            movement: 2,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Dash, S::Escape, S::Fortify, S::Amphibious],
            tribe_type: TribeType::Aquarion,
            ..Default::default()
        },
        U::Tridention => UnitSetting {
            cost: 8,
            attack: 2.5,
            movement: 2,
            defense: 1.0,
            range: 2,
            health: 10.0,
            skills: skills![S::Dash, S::Persist, S::Amphibious],
            tribe_type: TribeType::Aquarion,
            ..Default::default()
        },

        // Polaris
        U::Gaami => UnitSetting {
            cost: 10,
            is_super: true,
            attack: 4.0,
            movement: 1,
            defense: 3.0,
            range: 1,
            health: 30.0,
            skills: skills![S::AutoFreeze, S::FreezeArea, S::Static],
            tribe_type: TribeType::Polaris,
            ..Default::default()
        },
        U::IceArcher => UnitSetting {
            cost: 3,
            is_super: false,
            attack: 1.0,
            movement: 1,
            defense: 1.0,
            range: 2,
            health: 10.0,
            skills: skills![S::Dash, S::Freeze, S::Fortify, S::Stiff],
            tribe_type: TribeType::Polaris,
            ..Default::default()
        },
        U::IceFortress => UnitSetting {
            cost: 15,
            is_super: false,
            attack: 4.0,
            movement: 1,
            defense: 3.0,
            range: 2,
            health: 20.0,
            skills: skills![S::Skate, S::Scout, S::Escape, S::Static],
            tribe_type: TribeType::Polaris,
            ..Default::default()
        },
        U::Mooni => UnitSetting {
            cost: 5,
            is_super: false,
            attack: 0.0,
            movement: 1,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Skate, S::AutoFreeze, S::Stiff, S::Static],
            tribe_type: TribeType::Polaris,
            ..Default::default()
        },
        U::BattleSled => UnitSetting {
            cost: 5,
            is_super: false,
            attack: 3.0,
            movement: 2,
            defense: 2.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Escape, S::Dash, S::Skate],
            tribe_type: TribeType::Polaris,
            ..Default::default()
        },

        // Cymanti
        U::Mantis => UnitSetting {
            cost: 5,
            is_super: false,
            attack: 3.0,
            movement: 1,
            defense: 3.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Dash, S::Creep],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Centipede => UnitSetting {
            cost: 10,
            is_super: true,
            attack: 4.0,
            movement: 2,
            defense: 3.0,
            range: 1,
            health: 20.0,
            skills: skills![S::Dash, S::Creep, S::Eat, S::Static],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Segment => UnitSetting {
            cost: 10,
            attack: 2.0,
            movement: 1,
            defense: 2.0,
            range: 1,
            health: 10.0,
            skills: skills![
                S::Independent,
                S::Creep,
                S::Explode,
                S::Dash,
                S::Stiff,
                S::Static
            ],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Doomux => UnitSetting {
            cost: 10,
            attack: 3.5,
            movement: 3,
            defense: 2.0,
            range: 1,
            health: 20.0,
            skills: skills![S::Dash, S::Creep, S::Explode, S::Static],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Shaman => UnitSetting {
            cost: 10,
            attack: 1.0,
            movement: 1,
            defense: 1.0,
            range: 1,
            health: 10.0,
            skills: skills![S::Convert, S::Swarm, S::Static],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Kiton => UnitSetting {
            cost: 3,
            attack: 1.0,
            movement: 1,
            defense: 3.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Poison, S::Creep],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Hexapod => UnitSetting {
            cost: 3,
            attack: 3.0,
            movement: 2,
            defense: 1.0,
            range: 1,
            health: 5.0,
            skills: skills![S::Dash, S::Escape, S::Creep, S::Sneak],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Raychi => UnitSetting {
            cost: 5,
            attack: 3.0,
            movement: 3,
            defense: 2.0,
            range: 1,
            health: 10.0,
            is_unique: true,
            skills: skills![S::Dash, S::Water],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Boomchi => UnitSetting {
            cost: 5,
            attack: 3.0,
            movement: 2,
            defense: 3.0,
            range: 1,
            health: 10.0,
            is_unique: true,
            skills: skills![S::Dash, S::Explode, S::Stiff, S::Amphibious],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Phychi => UnitSetting {
            cost: 3,
            attack: 0.7,
            movement: 2,
            defense: 1.0,
            range: 2,
            health: 5.0,
            skills: skills![S::Dash, S::Fly, S::Poison, S::Surprise, S::DoubleAttack],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Exida => UnitSetting {
            cost: 8,
            attack: 3.0,
            movement: 1,
            defense: 1.0,
            range: 3,
            health: 10.0,
            skills: skills![S::Poison, S::Splash, S::Creep],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::LivingIsland => UnitSetting {
            cost: 20,
            attack: 4.0,
            movement: 2,
            defense: 4.0,
            range: 1,
            health: 20.0,
            is_super: false,
            skills: skills![S::Water, S::Algae, S::Stomp, S::Poison, S::Static],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::InsectEgg => UnitSetting {
            cost: 0,
            attack: 2.0,
            movement: 0,
            defense: 3.0,
            range: 1,
            health: 10.0,
            is_super: false,
            skills: skills![S::Grow, S::Explode, S::Static, S::Stiff],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Larva => UnitSetting {
            cost: 0,
            attack: 2.0,
            movement: 1,
            defense: 2.0,
            range: 1,
            health: 10.0,
            is_super: false,
            skills: skills![
                S::Dash,
                S::Creep,
                S::Surprise,
                S::Independent,
                S::Static,
                S::Grow
            ],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },
        U::Moth => UnitSetting {
            cost: 5,
            attack: 2.0,
            movement: 2,
            defense: 0.1,
            range: 1,
            health: 10.0,
            is_super: false,
            skills: skills![
                S::Fly,
                S::Dash,
                S::Sneak,
                S::Infiltrate,
                S::Poison,
                S::Static,
                S::Stiff
            ],
            tribe_type: TribeType::Cymanti,
            ..Default::default()
        },

        // Elyrion
        U::DragonEgg => UnitSetting {
            cost: 10,
            attack: -1.0,
            movement: 1,
            defense: 2.0,
            range: 1,
            health: 10.0,
            is_super: true,
            skills: skills![S::Grow, S::Fortify, S::Static, S::Stiff],
            tribe_type: TribeType::Elyrion,
            ..Default::default()
        },
        U::BabyDragon => UnitSetting {
            cost: 10,
            attack: 3.0,
            movement: 2,
            defense: 3.0,
            range: 1,
            health: 15.0,
            is_super: true,
            skills: skills![S::Grow, S::Dash, S::Fly, S::Escape, S::Scout, S::Static],
            tribe_type: TribeType::Elyrion,
            ..Default::default()
        },
        U::FireDragon => UnitSetting {
            cost: 10,
            attack: 4.0,
            movement: 3,
            defense: 3.0,
            range: 2,
            health: 20.0,
            is_super: true,
            skills: skills![S::Dash, S::Fly, S::Splash, S::Scout, S::Static],
            tribe_type: TribeType::Elyrion,
            ..Default::default()
        },
        U::Polytaur => UnitSetting {
            cost: 2,
            attack: 3.0,
            movement: 1,
            defense: 1.0,
            range: 1,
            health: 15.0,
            skills: skills![S::Dash, S::Fortify, S::Independent, S::Static],
            tribe_type: TribeType::Elyrion,
            ..Default::default()
        },
    }
}

/// Check if a unit type has a specific skill
pub fn has_skill(unit_type: UnitType, skill: SkillType) -> bool {
    get_unit_setting(unit_type).skills.contains(&skill)
}

/// Get super unit type for a tribe
pub fn get_super_unit(tribe_type: TribeType) -> UnitType {
    match tribe_type {
        TribeType::Polaris => UnitType::Gaami,
        TribeType::Aquarion => UnitType::Crab,
        TribeType::Elyrion => UnitType::DragonEgg,
        TribeType::Cymanti => UnitType::Centipede,
        _ => UnitType::Giant,
    }
}
