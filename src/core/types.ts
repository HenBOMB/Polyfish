export enum ModeType {
	Perfection = 0,
	Domination = 1,
}

export enum TerrainType {
	None 			= 0,
	Water 			= 1,
	Ocean 			= 2,
	Field 			= 3,
	Mountain		= 4,
	Forest 			= 5,
	Ice 			= 6,
	Capital			= 7,
}

export enum TechnologyType {
	None 			= 0,
	Riding 			= 1,
	FreeSpirit 		= 2, 
	Chivalry 		= 3,
	Roads 			= 4,
	Trade 			= 5,
	Organization 	= 6,
	Strategy 		= 7,
	Farming 		= 8,
	Construction 	= 9,
	Fishing 		= 10,
	Ramming			= 39, // previously aquaculture
	Aquatism 		= 12,
	Sailing 		= 13,
	Navigation 		= 14,
	Hunting 		= 15,
	Forestry 		= 16,
	Mathematics 	= 17,
	Archery 		= 18,
	Spiritualism 	= 19,
	Climbing 		= 20,
	Meditation 		= 21,
	Philosophy 		= 22,
	Mining 			= 23,
	Smithery 		= 24,
	FreeDiving 		= 25,
	Spearing 		= 26,
	Amphibian 		= 27,
	ForestMagic	    = 28,
	Frostwork		= 30,
	PolarWarfare	= 31,
	Polarism		= 32,
	ShockTactics    = 35,
	Recycling 		= 36,
	Hydrology 		= 37,
	Diplomacy		= 38,
	Sledding		= 41,
	IceFishing		= 42,
	Pascetism		= 43,
	Unbuildable 	= -1, 
	// TODO = Aquarion
}

export enum TribeType {
	None			= 0,
	Nature			= 1,
	AiMo			= 2,
	Aquarion		= 3,
	Bardur			= 4,
	Elyrion			= 5,
	Hoodrick		= 6,
	Imperius		= 7,
	Kickoo			= 8,
	Luxidoor		= 9,
	Oumaji			= 10,
	Quetzali		= 11,
	Vengir			= 12,
	XinXi			= 13,
	Zebasi			= 15,
	Yadakk			= 14,
	Polaris			= 16,
	Cymanti			= 17,
}

export enum ClimateType {
	Nature			= 0,
	XinXi			= 1,
	Imperius		= 2,
	Bardur			= 3,
	Oumaji			= 4,
	Kickoo			= 5,
	Hoodrick		= 6,
	Luxidoor		= 7,
	Vengir			= 8,
	Zebasi			= 9,
	AiMo			= 10,
	Aquarion		= 11,
	Quetzali		= 12,
	Elyrion			= 13,
	Yadakk			= 14,
	Polaris			= 15,
	Cymanti			= 16,
}

export enum RewardType {
	None 			 = 0,
	CityWall 		 = 1,
	Park 			 = 2,
	Workshop 		 = 3,
	Explorer 		 = 4,
	BorderGrowth 	 = 5,
	SuperUnit 		 = 6,
	Resources 		 = 7,
	PopulationGrowth = 8,
}

export enum UnitType {
	None       		= 0,
	// Scout         	= 1, // removed from game
	Warrior       	= 2,
	Rider         	= 3,
	Knight        	= 4,
	Defender      	= 5,
	// Ship          	= 6, // ? removed from game
	// Battleship    	= 7, // removed from game
	Catapult      	= 8,
	Archer        	= 9,
	MindBender    	= 10,
	Swordsman     	= 11,
	Giant        	= 12,
	Polytaur      	= 15,
	// Navalon      	= 16, // removed from game
	DragonEgg      	= 17,
	BabyDragon		= 18,
	FireDragon		= 19,
	Amphibian     	= 20,
	Tridention    	= 21,
	Mooni         	= 22,
	BattleSled    	= 23,
	IceFortress		= 24,
	IceArcher 		= 25,
	Crab          	= 26,
	Gaami       	= 27,
	Hexapod       	= 28,
	Doomux       	= 29,
	Phychi       	= 30,
	Kiton			= 31,
	Exida			= 32,
	Centipede		= 33,
	Segment        	= 34,
	Raychi        	= 35,
	Shaman        	= 36,
	Dagger         	= 37,
	Cloak         	= 38,
	Dinghy         	= 39,
	Pirate         	= 40,
	Bomber        	= 41,
	Scout     		= 42,
	Raft			= 43,
	Rammer    		= 44,
	Juggernaut		= 45,
}

export enum SkillType {
	None 			= 0,
	Dash          	= 1,
	Escape        	= 2,
	Scout         	= 3,
	Hide          	= 5,
	Build         	= 6,
	Persist       	= 7,
	Convert       	= 8,
	Heal          	= 9,
	Float         	= 10,
	Carry         	= 11,
	Grow          	= 12,
	Fly           	= 13,
	Splash        	= 14,
	Decay         	= 15,
	Navigate      	= 16,
	Crush         	= 17,
	Freeze        	= 18,
	FreezeArea    	= 19,
	AutoFreeze    	= 20,
	Skate         	= 21,
	Fortify       	= 22,
	Creep         	= 23,
	Boost         	= 24,
	Independent   	= 25,
	Poison        	= 26,
	Eat           	= 27,
	Unique          = 29, 
	Explode			= 30,
	Surprise		= 31,
	Agent			= 32, // ! This unit can be built on enemy terrirory?
	Infiltrate		= 35,
	Detect			= 36, // This unit can detect nearby invisible units
	Intercept		= 37, 
	Stiff           = 38, // Cannot retaliate
	Protect         = 39, // No description
	Stomp           = 40, // Damage all surrounding units when moving
	AutoHeal        = 41,
	Static        	= 42, // ! Prevents a unit from becoming a veteran
	AutoFlood		= 79, // Unique to crab
}

export enum AbilityType {
	None = 0,
	// Burn Forest is an ability that turns a forest tile into a field tile with crop at a cost of two stars. 
	// This ability is unlocked by the Construction technology. It can be used on any forest tile in the player's territory that does not have a building on it. 
	// If the ability is used on a tile with a wild animal, the wild animal will be forfeited.
	// This ability can be used with the Grow Forest ability and Farming to produce population from an empty field tile by growing a forest, burning the forest, and finally building a Farm.
	// The ∑∫ỹriȱŋ and Cymanti tribes do not have access to this ability. 
	// Costs 5, Effect: Turns a forest tile into a field tile with crop
	BurnForest 		= 1,
	// Clear Forest is an ability that turns a forest tile into a field tile and gives the player one star. 
	// The ability is unlocked by the Forestry technology and can only be used on forest tiles in the player's territory without any buildings or structures on them. 
	// If the ability is used on a tile with a wild animal, the wild animal will be forfeited.
	// The ∑∫ỹriȱŋ tribe does not have access to this ability. 
	ClearForest 	= 2,
	// Grow Forest is an ability that turns a field tile into a forest tile at a cost of five stars.
	// The ability is unlocked by the Spiritualism technology and can only be used on field tiles in the player's territory without any buildings on them. 
	// The Grow Forest ability can be used to build a Forest Temple on a field tile. Also, it can be used with the Burn Forest ability and Farming to produce population from an empty field tile by growing then burning a forest and finally building a Farm. In addition, Grow Forest has a limited military use; because forests hinder movement, they can be used to frustrate invading enemies. 
	GrowForest 		= 3,
	// Destroy is an ability that removes a building or ruin at no cost. 
	// The ability is unlocked by the Chivalry technology.
	// Destroying a building also removes any population or points provided by the building. 
	// Destroyed monuments and ice banks cannot be rebuilt.
	// Destroy can be used in "scorched-earth" tactics before a city is captured by an enemy. 
	Destroy 		= 4,
	// The Cymanti tribe has the Decompose ability instead of Destroy. 
	// The two abilities are identical except that the Decompose ability removes the building or ruin at the end of the turn (instead of immediately) and returns a building's full cost. 
	Decompose 		= 5,
	
	// Starfish Harvesting is an ability that can only be used on a tile with a Starfish, removes the Starfish from the tile, and gives the player 8 stars. The ability is unlocked by the Navigation technology. 
	// Moving any naval unit onto a Starfish, even in neutral or enemy territory, will allow it to collect 8 stars by removing the Starfish the next turn. This consumes the unit's turn, much like if it were examining a ruin. 
	// Handles by Capture move and Navigation tech
	// TODO not sure if other special tribes CAN NOT capture / harvest starfish
	// StarfishHarvesting = 6,

	/**
	 * Heals a unit by up to 4 HP in friendly territory and 2 HP elsewhere. (Cannot heal past maximum health.) 
	 * Removes poison and heals 0 HP if the unit is poisoned. 
	 * Available to all units below their maximum health.
	 * Note: If a unit does not take any action in a given turn, it will automatically heal if it can do so. 
	 */
	Recover 		= 7,
	/**
	 * Removes a unit and refunds half its cost, rounded down.
	 * Available to all units once Free Spirit has been researched.
	 * Disband can only be used by units that have not performed any action
	 * Disbanding a super unit returns five stars. (Super units have a theoretical cost of 10 stars.) 
	 * Disbanding a Ship or Battleship only returns half the cost of the unit it carries. 
	 * No portion of the cost of the upgrade from a Boat is returned. 
	 */
	Disband 		= 8,
	/**
	 * Causes a ship unit to turn into the unit it is carrying.
	 * Available to any naval unit that is carrying another unit and is occupying a flooded tile. 
	 */
	// Disembark = 3,
	/**
	 * Heals all adjacent friendly units by up to 4 HP. (Cannot heal past maximum health.) 
	 * Only available to units with the heal skill (i.e., the Mind Bender). ASSUMING IT CLEARS POISON BEFORE HEAL
	 */
	HealOthers 		= 9,
	/**
	 * Breaks all adjacent ice tiles into what they were before they were frozen.
	 * Can only be used when there is at least one adjacent ice tile.
	 * Available to all units except those with the freeze area skill (i.e., the Mooni and the Gaami).
	 */
	BreakIce 		= 10, 
	/**
	 * Turns a flooded tile into an unflooded tile.
	 * Available to any unit occupying a flooded tile (even ships).
	 */
	Drain 			= 11,
	/**
	 * Floods the tile that a unit is on. Can only be used on unflooded fields and forests.
	 * Only available to Aquarion units once the Waterways technology has been researched.
	 */
	// FloodTile 		= 12,
	/**
	 * Freezes all adjacent water and ocean tiles into ice and converts all adjacent land tiles into Polaris land (if and only if the unit belongs to the Polaris tribe). Also freezes all adjacent units.
	 * Only available to units with the freeze area skill (i.e., the Mooni and the Gaami).
	 */
	FreezeArea 		= 13,
	/**
	 * Boosts all adjacent friendly units, increasing their attack by 0.5 and movement by 1 until their next action (not including moving).
	 * Only available to units with the boost skill (i.e., the Shaman).
	 */
	Boost 			= 14,
	/**
	 * Poisons and damages (the same amount as if it was an individual attack) all adjacent enemy units.
	 * Removes the unit and leaves spores (on land) or Algae (in water) in its place.
	 * Only available to units with the explode skill (i.e., the Segment, Raychi, and Doomux).
	 */
	Explode 		= 15,

	Promote 		= 16,
}

export enum StructureType {
	None            = 0,
	Village         = 1,
	Ruin            = 2,
	Farm            = 5,
	Windmill        = 6,
	Port            = 8,
	LumberHut       = 12,
	Sawmill         = 13,
	Temple          = 17,
	ForestTemple    = 18,
	WaterTemple     = 19,
	MountainTemple  = 20,
	Mine            = 21,
	Forge           = 22,
	AltarOfPeace    = 23,
	TowerOfWisdom   = 24,
	GrandBazaar     = 25,
	EmperorsTomb    = 26,
	GateOfPower     = 27,
	ParkOfFortune   = 28,
	EyeOfGod        = 29,
	Outpost         = 33,
	Spores          = 37,
	Swamp           = 38,
	Mycelium        = 39,
	Lighthouse      = 47,
	Bridge          = 48,
	Market          = 50,
	IceTemple       = 69, // TODO find id
	Embassy         = 70, // TODO find id
	Road         	= 71, // TODO find id
}

export enum ResourceType {
	None 		    = 0,
	WildAnimal 		= 1,
	Crop 			= 2,
	Fish 			= 3,
	Unknown1        = 4,
	Metal 			= 5,
	Fruit 			= 6,
	Spores 			= 7,
	Starfish 		= 8,
}

export enum EffectType {
	None 		= 0,
	Poison		= 1,
	Boost		= 2,
	Invisible	= 3,
	Frozen		= 10, // TODO ID does not match live polytopia game, fix
}

export enum CaptureType {
	None 	 = 0,
	Ruins 	 = 1,
	Starfish = 2,
	Village  = 3,
	City 	 = 4,
}

export enum MoveType {
    None 	 = 0,
    Step 	 = 1,
    Attack   = 2,
    Ability  = 3,
    Summon   = 4,
    Harvest  = 5,
    Build    = 6,
    Research = 7,
    Capture  = 8,
    Reward   = 9,
    EndTurn  = 10
}

export enum TaskType {
	/**
	 * meditation, dont attack for 5 turns
	 */
    Pacifist   = 0,
	/**
	 * philosophy, unlock all tech
	 */
    Genius 	   = 1,
	/**
	 * trade, have 100 stars
	 */
    Wealth     = 2,
	/**
	 * discover every lighthouse
	 */
    Explorer   = 3,
	/**
	 * kill 10 units
	 */
    Killer     = 4,
	/**
	 * connect 5 cities to capital
	 */
    Network    = 5,
	/**
	 * lvl 5 city
	 */
    Metropolis = 6,
}

export const EconomyAbilityTypes: AbilityType[] = [
	AbilityType.BreakIce, AbilityType.BurnForest, AbilityType.ClearForest, 
	AbilityType.Decompose, AbilityType.Destroy, AbilityType.Drain, 
	AbilityType.GrowForest
] as any;

export const ArmyAbilityTypes: AbilityType[] = [
	AbilityType.Boost, AbilityType.Disband, AbilityType.Explode, 
	AbilityType.FreezeArea, AbilityType.HealOthers, AbilityType.Recover, 
	AbilityType.Promote
] as any;

Object.freeze(EconomyAbilityTypes);
Object.freeze(ArmyAbilityTypes);