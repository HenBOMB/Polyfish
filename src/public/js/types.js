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
    3: "Land",
    4: "Mountain",
    5: "Forest",
    6: "Ice",
    7: "GroundWater",
}

const RewardType = {
    "Workshop": 3,
    "Park": 2,
}

const TechnologyNames = {
    1: "Riding", 2: "Free Spirit", 3: "Chivalry", 4: "Roads", 5: "Trade",
    6: "Organization", 7: "Strategy", 8: "Farming", 9: "Construction",
    10: "Fishing", 12: "Aquatism", 13: "Sailing", 14: "Navigation",
    15: "Hunting", 16: "Forestry", 17: "Mathematics", 18: "Archery", 19: "Spiritualism",
    20: "Climbing", 21: "Meditation", 22: "Philosophy", 23: "Mining", 24: "Smithery",
    30: "Frostwork", 31: "Polar Warfare", 32: "Polarism", 36: "Recycling", 37: "Hydrology"
}

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
    8: [9],       // Farming -> Construction
    // Climbing branch
    20: [23, 21], // Climbing -> Mining, Meditation
    23: [24],     // Mining -> Smithery
    21: [22],     // Meditation -> Philosophy
    // Fishing branch
    10: [13, 11], // Fishing -> Sailing, Ramming (11 is placeholder for Ramming if needed)
    13: [14],     // Sailing -> Navigation
    // Hunting branch
    15: [18, 16], // Hunting -> Archery, Forestry
    18: [19],     // Archery -> Spiritualism
    16: [17],     // Forestry -> Mathematics
};


const StructureNames = {
    1: "Village", 2: "Ruin", 5: "Farm", 6: "Windmill", 8: "Port",
    12: "Lumber Hut", 13: "Sawmill", 17: "Temple", 18: "Forest Temple",
    19: "Water Temple", 20: "Mountain Temple", 21: "Mine", 22: "Forge",
    23: "Altar of Peace", 24: "Tower of Wisdom", 25: "Grand Bazaar",
    26: "Emperor's Tomb", 27: "Gate of Power", 28: "Park of Fortune", 29: "Eye of God",
    33: "Outpost", 37: "Spores", 38: "Swamp", 39: "Mycelium", 40: "Algae",
    47: "Lighthouse", 48: "Bridge", 50: "Market", 69: "Ice Temple",
    70: "Embassy", 71: "Road"
};

const ResourceType = {
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
