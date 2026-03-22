const CLIMATE_IDS = [
    "Nature", "Xin-xi", "Imperius", "Bardur", "Oumaji",
    "Kickoo", "Hoodrick", "Luxidoor", "Vengir",
    "Zebasi", "Ai-Mo", "Aquarion", "Quetzali",
    "∑∫ỹriȱŋ", "Yădakk", "Polaris", "Cymanti"
];

const TRIBE_ID_2_NAME = [
    null, 'Nature', 'Ai-Mo', 'Aquarion', 'Bardur',
    '∑∫ỹriȱŋ', 'Hoodrick', 'Imperius', 'Kickoo',
    'Luxidoor', 'Oumaji', 'Quetzali', 'Vengir',
    'Xin-xi', 'Yădakk', 'Zebasi', 'Polaris', 'Cymanti'
];

const TRIBE_COLORS = {
    1: '#0ea5e9', // Nature (Blue-ish)
    2: '#8b5cf6', // Ai-Mo (Purple)
    3: '#3b82f6', // Aquarion (Blue)
    4: '#1f2937', // Bardur (Black/Grey)
    5: '#ec4899', // ∑∫ỹriȱŋ (Pink)
    6: '#4ade80', // Hoodrick (Green)
    7: '#0ea5e9', // Imperius (Blue)
    8: '#22c55e', // Kickoo (Green)
    9: '#fbbf24', // Luxidoor (Gold/Yellow)
    10: '#f59e0b', // Oumaji (Yellow/Orange)
    11: '#22c55e', // Quetzali (Green)
    12: '#475569', // Vengir (Dark Grey)
    13: '#ef4444', // Xin-xi (Red)
    14: '#f97316', // Yădakk (Orange)
    15: '#facc15', // Zebasi (Yellow)
    16: '#38bdf8', // Polaris (Light Blue)
    17: '#16a34a', // Cymanti (Green)
};

const ClassNameToId = {
    1: "Scout",
    2: "Warrior",
    3: "Rider",
    4: "Knight",
    5: "Defender",
    7: "Battleship",
    8: "Catapult",
    9: "Archer",
    10: "MindBender",
    11: "Swordsman",
    12: "Giant",
    15: "Polytaur",
    20: "Amphibian",
    21: "Tridention",
    22: `Mooni`,
    23: "BattleSled",
    25: "IceArcher",
    26: "Crab",
    28: "Hexapod",
    31: "Kiton",
    35: "Raychi",
    36: "Shaman",
    38: "Cloak",
    39: "Cloak_Boat",
    41: "Bombership",
    42: "Scoutship",
    43: "Boat",
    44: "Rammership",
    45: "Juggernaut",
};

const CLIMATE_TO_ANIMAL = [
    'Invalid',
    'horse0001', // xinxi,
    'horse0002', // imperius,
    'horse0003', // bardur,
    'horse0004', // oumaji,
    'horse0005', // kickoo,
    'horse0006', // hoodrick,
    'horse0007', // luxidoor,
    'horse0008', // vengir,
    'horse0009', // zebasi,
    'horse0010', // aimo,
    'horse0011', // aquarion,
    'horse0012', // quetzali,
    'horse0013', // '∑∫ỹriȱŋ',
    'horse0014', // yadakk,
    'animal_15', // polaris,
    'lytheti', // cymanti
]

const OWNER_TO_ID_INDEX = [
    null, 0, 10, 11, 3, 13, 6, 2, 5, 7, 4, 12, 8, 1, 14, 9, 15, 16
]

const TerrainType = {
    0: "None",
    1: "Water",
    2: "Ocean",
    3: "Field",
    4: "Mountain",
    5: "Forest",
    6: "Ice",
    7: "Wetland",
    8: "Mangrove",
}

const RewardTypes = {
    "CityWall": 1,
    "Park": 2,
    "Workshop": 3,
    "Explorer": 4,
    "BorderGrowth": 5,
    "SuperUnit": 6,
    "Resources": 7,
    "PopGrowth": 8,
    1: "CityWall",
    2: "Park",
    3: "Workshop",
    4: "Explorer",
    5: "BorderGrowth",
    6: "SuperUnit",
    7: "Resources",
    8: "PopGrowth",
}

const RewardEmojis = {
    0: "none",
    1: "🛡️",
    2: "🌲",
    3: "⚒️",
    4: "☁️",
    5: "🧾",
    6: "🎖️",
    7: "⭐",
    8: "😄",
}

const TechnologyNames = {
    1: "Riding", 2: "Free Spirit", 3: "Chivalry", 4: "Roads", 5: "Trade",
    6: "Organization", 7: "Strategy", 8: "Farming", 9: "Construction",
    10: "Fishing", 12: "Aquatism", 13: "Sailing", 14: "Navigation",
    15: "Hunting", 16: "Forestry", 17: "Mathematics", 18: "Archery", 19: "Spiritualism",
    20: "Climbing", 21: "Meditation", 22: "Philosophy", 23: "Mining", 24: "Smithery",
    25: "Free Diving", 26: "Spearing", 28: "Waterways", 30: "Frostwork", 31: "Polar Warfare",
    32: "Polarism", 35: "Shock Tactics", 36: "Recycling", 38: "Diplomacy", 39: "Ramming",
    41: "Sledding", 42: "Ice Fishing", 43: "Pascetism", 49: "Oceantology", 121: "Rituals"
}

// Tech tree dependencies: tech_id -> [next techs that require it]
// Tech tree dependencies: tech_id -> [next techs that require it]
const TechTree = {
    // Unrequired (implicit) -> Tier 1 roots
    0: [1, 6, 20, 10, 15], // Riding, Organization, Climbing, Fishing, Hunting
    // Riding branch
    1: [2, 4],    // Riding -> Free Spirit, Roads
    4: [5],       // Roads -> Trade
    2: [3],       // Free Spirit -> Chivalry
    // Organization branch
    6: [7, 8],    // Organization -> Strategy, Farming
    7: [38],      // Strategy -> Diplomacy
    8: [9],       // Farming -> Construction
    // Climbing branch
    20: [23, 21], // Climbing -> Mining, Meditation
    23: [24],     // Mining -> Smithery
    21: [22],     // Meditation -> Philosophy
    // Fishing branch
    10: [13, 39], // Fishing -> Sailing, Ramming
    13: [14],     // Sailing -> Navigation
    39: [12],     // Ramming -> Aquatism
    // Hunting branch
    15: [18, 16], // Hunting -> Archery, Forestry
    18: [19],     // Archery -> Spiritualism
    16: [17],     // Forestry -> Mathematics
};

// Visual Nodes Configuration (Radial Layout - 72deg separation)
// Branch Angles: Org(0), Climbing(72), Fishing(144), Hunting(216/-144), Riding(288/-72)
const TechNodes = {
    0: { x: 0, y: 0, icon: "🏛️" }, // Start

    // Organization Branch (0 deg - East)
    6: { x: 160, y: 0, icon: "🤲" }, // Organization
    8: { x: 260, y: 30, icon: "🌾" }, // Farming
    9: { x: 360, y: 60, icon: "🏗️" }, // Construction
    7: { x: 260, y: -30, icon: "🛡️" }, // Strategy
    38: { x: 360, y: -60, icon: "🤝" }, // Diplomacy

    // Climbing Branch (72 deg - South East)
    20: { x: 60, y: 160, icon: "🧗" }, // Climbing
    23: { x: 100, y: 240, icon: "⛏️" }, // Mining
    24: { x: 140, y: 320, icon: "⚔️" }, // Smithery
    21: { x: 0, y: 240, icon: "🧘" }, // Meditation
    22: { x: -20, y: 340, icon: "🦉" }, // Philosophy

    // Fishing Branch (144 deg - South West)
    10: { x: -130, y: 100, icon: "🎣" }, // Fishing
    13: { x: -240, y: 120, icon: "⛵" }, // Sailing
    14: { x: -340, y: 140, icon: "🧭" }, // Navigation
    12: { x: -200, y: 200, icon: "🐢" }, // Aquatism
    39: { x: -280, y: 280, icon: "🐏" }, // Ramming

    // Hunting Branch (-144 deg - North West)
    15: { x: -130, y: -100, icon: "🏹" }, // Hunting
    18: { x: -240, y: -120, icon: "🎯" }, // Archery
    19: { x: -340, y: -140, icon: "👻" }, // Spiritualism
    16: { x: -200, y: -200, icon: "🌲" }, // Forestry
    17: { x: -280, y: -280, icon: "📐" }, // Mathematics

    // Riding Branch (-72 deg - North East)
    1: { x: 60, y: -160, icon: "🏇" }, // Riding
    4: { x: 140, y: -250, icon: "🛣️" }, // Roads
    5: { x: 200, y: -340, icon: "💰" }, // Trade (Far out)
    2: { x: 0, y: -240, icon: "🕊️" }, // Free Spirit
    3: { x: -40, y: -320, icon: "⚔️" }, // Chivalry
};

const StructureCosts = {
    5: 5, // Farm
    6: 5, // Windmill
    8: 10, // Port
    12: 3, // Lumber Hut
    13: 5, // Sawmill
    17: 20, // Temple
    18: 20, // Forest Temple
    19: 20, // Water Temple
    20: 20, // Mountain Temple
    21: 5, // Mine
    22: 5, // Forge
    25: 5, // Grand Bazaar
    33: 5, // Outpost
    50: 5, // Market
    69: 20, // Ice Temple
    71: 3, // Road
};

const StructureTypes = {
    1: "Village", 2: "Ruin", 3: "Road", 5: "Farm", 6: "Windmill", 8: "Port",
    12: "Lumber Hut", 13: "Sawmill", 17: "Temple", 18: "Forest Temple",
    19: "Water Temple", 20: "Mountain Temple", 21: "Mine", 22: "Forge",
    23: "Altar of Peace", 24: "Tower of Wisdom", 25: "Grand Bazaar",
    26: "Emperor's Tomb", 27: "Gate of Power", 28: "Park of Fortune", 29: "Eye of God",
    32: "Sanctuary", 33: "Outpost", 34: "Ice Bank", 35: "Ice Temple",
    37: "Fungi", 38: "Algae", 39: "Mycelium", 41: "Clathrus", 47: "Lighthouse",
    48: "Bridge", 50: "Market", 70: "Embassy"
};

const StructureEmojis = {
    3: "🛣️",
    5: "🌾",
    6: "🌬️",
    8: "⚓",
    12: "🪵",
    13: "🪚",
    17: "⛩️",
    18: "🌲",
    19: "🌊",
    20: "⛰️",
    21: "⛏️",
    22: "⚒️",
    23: "☮️",
    24: "🦉",
    25: "🏪",
    26: "⚰️",
    27: "⚡",
    28: "🏆",
    29: "👁️",
    32: "⛩️",
    33: "🏰",
    34: "💰",
    35: "🧊",
    37: "🍄",
    38: "🌿",
    39: "🧶",
    41: "🕸️",
    47: "🗼",
    48: "🌉",
    50: "💰",
    70: "🏛️",
};

const ResourceTypes = {
    0: "None",
    1: "Game",
    2: "Crop",
    3: "Fish",
    5: "Metal",
    6: "Fruit",
    7: "Spores",
    8: "Starfish",
    9: "AquaCrop",
};

const ResourceEmojis = {
    0: "❓",
    1: "🍖",
    2: "🌾",
    3: "🐟",
    5: "⛏️",
    6: "🥝",
    7: "🍄",
    8: "⭐",
    9: "🌱",
};


const UnitTypes = {
    0: "None",
    2: "Warrior",
    3: "Rider",
    4: "Knight",
    5: "Defender",
    8: "Catapult",
    9: "Archer",
    10: "MindBender",
    11: "Swordsman",
    12: "Giant",
    15: "Polytaur",
    17: "DragonEgg",
    18: "BabyDragon",
    19: "FireDragon",
    20: "Amphibian",
    21: "Tridention",
    22: "Mooni",
    23: "BattleSled",
    24: "IceFortress",
    25: "IceArcher",
    26: "Crab",
    27: "Gaami",
    28: "Hexapod",
    29: "Doomux",
    30: "Phychi",
    31: "Kiton",
    32: "Exida",
    33: "Centipede",
    34: "Segment",
    35: "Raychi",
    36: "Shaman",
    37: "Dagger",
    38: "Cloak",
    39: "Dinghy",
    40: "Pirate",
    41: "Bomber",
    42: "Scout",
    43: "Raft",
    44: "Rammer",
    45: "Juggernaut",

    120: "LivingIsland",
    121: "Boomchi",
    122: "Moth",
    123: "Larva",
    124: "InsectEgg",
    125: "Mantis",
};

const UnitEmojis = {
    2: "🪖", // Warrior
    3: "🏇", // Rider
    4: "🤺", // Knight
    5: "🛡️", // Defender
    8: "☄️", // Catapult
    9: "🏹", // Archer
    10: "🧙", // MindBender
    11: "🗡️", // Swordsman
    12: "👹", // Giant
    15: "🐮", // Polytaur
    20: "🐸", // Amphibian
    21: "🔱", // Tridention
    22: "❄️", // Mooni
    23: "🛷", // BattleSled
    24: "🏰", // IceFortress
    25: "🏹", // IceArcher
    28: "🕷️", // Hexapod
    29: "🪲", // Doomux
    30: "🦟", // Phychi
    31: "🐚", // Kiton
    32: "🐝", // Exida
    33: "🐛", // Centipede
    36: "🪄", // Shaman
    121: "🧨", // Boomchi
};

const CaptureTypes = {
    0: 'None',
    1: 'Ruins',
    2: 'Starfish',
    3: 'Village',
    4: 'City',
}

const AbilityType = {
    None: 0,
    BurnForest: 1,
    ClearForest: 2,
    GrowForest: 3,
    Destroy: 4,
    Decompose: 5,
    Convert: 6,
    Recover: 7,
    Disband: 8,
    HealOthers: 9,
    Drain: 11,
    FreezeArea: 13,
    Boost: 14,
    Explode: 15,
    Promote: 16,
    BreakIce: 17,
    BreakPeace: 18,
    EnchantAnimal: 23,
};

const AbilityNames = {
    1: "Burn Forest",
    2: "Clear Forest",
    3: "Grow Forest",
    4: "Destroy",
    5: "Decompose",
    6: "Convert",
    7: "Recover",
    8: "Disband",
    9: "Heal Others",
    11: "Drain",
    13: "Freeze Area",
    14: "Boost",
    15: "Explode",
    16: "Promote",
    17: "Break Ice",
    18: "Break Peace",
    23: "Enchant Animal",
};

const AbilityEmojis = {
    1: "🔥",
    2: "🪓",
    3: "🌿",
    4: "💣",
    5: "♻️",
    6: "🌀",
    7: "🩹",
    8: "👋",
    9: "💖",
    11: "🩸",
    13: "🧊",
    14: "⚡",
    15: "💥",
    16: "🎖️",
    17: "⛏️",
    18: "💔",
    23: "🦄",
};

const AbilityCosts = {
    1: 2, // Burn Forest
    2: -1, // Clear Forest (Gains 1 star)
    3: 5, // Grow Forest
    4: 0, // Destroy
    5: 0, // Decompose
    6: 0, // Convert
    7: 0, // Recover
    8: 0, // Disband
    9: 0, // Heal Others
    11: 0, // Drain
    13: 0, // Freeze Area
    14: 0, // Boost
    15: 0, // Explode
    16: 0, // Promote
    17: 0, // Break Ice
    18: 0, // Break Peace
    23: 3, // Enchant Animal
};
