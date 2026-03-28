use crate::states::GameState;

// enum
pub enum GameVersion {
    AquarionRework = 105,
    TheForgotten = 108,
    BalancePass2025 = 114,
    /*
        Cymanti Rework!
        Full Changelog
    Naval changes
    – Cymanti explorer can now move on water like other tribes
    – Moved around naval techs to replicate Path of the Ocean changes
    – Algae is now a tile effect instead of an improvement, other improvements can be built on top of it. Spawns with a fruit on top
    – Algae does not cripple movement for naval units (but still for land units)
    – Clathrus is now a land building and grows from spores instead of algae. It is poisonous, but no longer creates networks (Mycelium can now be built on Algae instead)
    – Raychi is split into two units; Boomchi that explodes to create algae, and a slightly adjusted Raychi
    – Boomchi get “amphibious” ability so they can move on land
    – ’Floating Island’ unit in navigation, spawns algae + area damage

    Other changes
    – ’Mantis’ replaces Swordsman, same stats but with ‘creep’ ability
    – Fungi is limited to level 2
    – ’Microbes’ ability in ‘Recycling’ allows Fungi to grow 1 additional level
    – Growing units (centipede head, dragons, larva etc.) inherit the damage from their previous state
    – ’Kiton’ get ‘creep’
    – Creep no longer overcomes mountain movement penalty
    – Phychi get a “double attack” ability
    – Poison defense debuff increased to 50% but defense bonuses are intact
    – Poison now slow down enemy units
    – Explode units can explode even after attacking
    – Doomux cannot become veteran
    – Doomux attack down to 3.5

    Diplomacy changes
    – Cloaks replaced by Moths, they can fly but are not invisible. When they infiltrate a city they poison defenses and lay Eggs
    – Eggs can’t move but can explode, and if you wait 1 tur,n they will hatch into Larva that are similar to Daggers
    – After 3 turns, Larva will turn into Moths, making the circle complete

    Shaman changes
    – ‘Boost’-ability replaced by ‘Swarm’ that only buffs movement
    – Movement buff is retained until the unit gets attacked
    – ’Shaman moved to ‘meditation’ so it is easier to replace
    – ’Mountain temple’ moved to ‘Philosophy’
    – ‘Pacifist’ task is replaced by ‘Converter’ (convert 3 enemies to build ‘Church of Converts’ monument)
     */
    CymantiRework = 115,
}

pub fn get_burn_forest_cost(gs: &GameState) -> i32 {
    if gs.settings.version <= GameVersion::AquarionRework as i32 {
        4
    } else {
        3
    }
}

pub fn get_clear_forest_stars(gs: &GameState) -> i32 {
    if gs.settings.version <= GameVersion::AquarionRework as i32 {
        2
    } else {
        1
    }
}

pub fn get_polytaur_cost(gs: &GameState) -> i32 {
    if gs.settings.version <= GameVersion::AquarionRework as i32 {
        2
    } else {
        1
    }
}
