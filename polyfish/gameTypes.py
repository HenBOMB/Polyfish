# Categorical
MOVE_TYPE = {
    'None':     -1,
    'Attack':    0,
    'Step':      1,
    'Capture':   2,
    'Ability':   3,
    'Summon':    4,
    'Harvest':   5,
    'Build':     6,
    'Research':  7,
    'Reward':    8,
    'EndTurn':   9,
    '_MAX_N':   10,
}

# Categorical
BUILD_TYPE = {
    # Unbuildable
    'None':          -1,
	'Village':       -1,
	'Ruin':          -1,
	'Mycelium':      -1,
	'Lighthouse':    -1,
	'Swamp':         -1,
	'Spores':        -1,
	# Riding
    'Road':           0,
	'Bridge':         1,
	'Temple':         3,
	'Market':         2,
    # Organization
	'Farm':           4,
	'Windmill':       5,
	'Embassy':        6,
    # Climbing
    'Mine':           7,
    'Forge':          9,
	'MountainTemple': 8,
    # Fishing
    'Port':          10,
	'WaterTemple':   11,
    # Hunting
	'LumberHut':     12,
	'Sawmill':       13,
	'ForestTemple':  14,
    # Unique
	'IcePort':       15,
	'IceTemple':     16,
	'AltarOfPeace':  17,
	'TowerOfWisdom': 18,
	'GrandBazaar':   19,
	'EmperorsTomb':  20,
	'GateOfPower':   21,
	'ParkOfFortune': 22,
	'EyeOfGod':      23,
    '_MAX_N':        24,
}

# Spatial, some use "to" (unit related), and others (not unit related), use "from"
ABILITY_TYPE = {
    'None':              -1,
    # Only require "from"
	'Recover':            0,
	'Boost':              1,
	'HealOthers':         2,
	'Explode':            3,
    # Only require "to"
    # Riding
	'Disband':            4,
	'Destroy':            5,
	'Decompose':          6,
    # Organization
	'BurnForest':         7,
    # Fishing
	'StarfishHarvesting': 8,
    # Hunting
	'GrowForest':         9,
	'ClearForest':       10,
    # Unique
	'BreakIce':          11, 
	'Drain':             12,
	'FloodTile':         13,
	'FreezeArea':        14,
    '_MAX_N':            15,
}

# Categorical
SUMMON_TYPE = {
    # Not spawnable / upgradable units
	'None':        -1,
	'Raft':        -1,
	'Pirate':      -1,
	'Dagger':      -1,
	'Giant':       -1,
    'Polytaur':    -1,
	'DragonEgg':   -1,
	'BabyDragon':  -1,
	'FireDragon':  -1,
	'Dinghy':      -1,
	'Crab':        -1,
	'Gaami':       -1,
	'Centipede':   -1,
	'Segment':     -1,
    # Basic
	'Warrior':      0,
	'Rider':        1, # Aquarion Amphibian, Cymanti Hexapod
	'Knight':       2, # Aquarion Tridention, Cymanti Doomux
	'Defender':     3, # Cymanti Kiton
	'Cloak':        4,
	'Swordsman':    5,
	'Archer':       6, # Polaris IceArcher, Cymanti Phychi
	'Catapult':     7, # Cymanti Exida
	'MindBender':   8, # Cymanti Shaman
	'Mooni':        9, # Unique unit, Polaris Fishing
	'BattleSled':  10, # Unique unit, Polaris Sledding -> Sailing
	'IceFortress': 11, # Unique unit, Polaris Frostwork -> Navigation
	'Raychi':      12, # Unique unit, Cymanti Sailing
    # Navy
	'Scout':       13,
	'Rammer':      14,
	'Bomber':      15,
	'Juggernaut':  16,
    '_MAX_N':      17,
}

# Categorical
TECHNOLOGY_TYPE = {
    'None':        -1,
    'Unbuildable': -1,
    # Riding
    'Riding':        0,
    'Roads':         1,
    'Trade':         2,
    'FreeSpirit':    3,
    'Chivalry':      4,
    # Organization 
    'Organization':  5,
    'Farming':       6,
    'Construction':  7,
    'Strategy':      8,
    'Diplomacy':     9,
    # Climbing
    'Climbing':     10,
    'Mining':       11,
    'Smithery':     12,
    'Meditation':   13,
    'Philosophy':   14,
    # Fishing
    'Fishing':      15,
    'Sailing':      16,
    'Navigation':   17,
    'Aquaculture':  18,
    'Aquatism':     19,
    # Hunting
    'Hunting':      20,
    'Archery':      23,
    'Spiritualism': 24,
    'Forestry':     21,
    'Mathematics':  22,
    # Unique
    '_MAX_N':       25,
}

# Sorted by economy / army
REWARD_TYPE = {
    'None':            -1,
    'Workshop':         0,
    'Resources':        1,
    'SuperUnit':        2,
    'PopulationGrowth': 3,
    'Explorer':         4,
    'CityWall':         5,
    'BorderGrowth':     6,
    'Park':             7,
    '_MAX_N':           8,
}
