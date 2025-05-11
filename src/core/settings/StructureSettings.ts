import { TerrainType, ResourceType, StructureType, TribeType } from "../types";

// ! Sorted by tier and clockwise per branch
export const StructureSettings: Record<StructureType, { 
    cost?: number, 
    terrainType?: Set<TerrainType>, 
    adjacentTypes?: Set<StructureType>, 
    resourceType?: ResourceType, 
    limitedPerCity?: true,
    rewardPop?: number;
    rewardStars?: number;
    tribeType?: TribeType;
}> = {
    [StructureType.None]: {
    },
    [StructureType.Village]: {
    },
    [StructureType.Ruin]: {
    },
    // TODO not sure how this works but, i think harvesting spores leads to swamp? or sumthin
    [StructureType.Spores]: {
        resourceType: ResourceType.Spores,
        terrainType: new Set([TerrainType.Forest]),
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Swamp]: {
        // TODO Spores
        // techType: TechnologyType.None,
        terrainType: new Set([TerrainType.Ocean]),
    },
    // TODO Mycelium
    [StructureType.Mycelium]: {
        // techType: TechnologyType.None,
        terrainType: new Set([TerrainType.Field]),
        limitedPerCity: true,
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Lighthouse]: {
    },

    // Riding
    [StructureType.Road]: {
        cost: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest],) 
    },
    [StructureType.Bridge]: {
        terrainType: new Set([TerrainType.Water]),
        // techType: TechnologyType.Roads,
        // terrainType: new Set(TerrainType.Water),
        // TODO Bridge logic for connected adjacent ground tiles or whatever
    },
    [StructureType.Temple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Field]),
    },
    [StructureType.Market]: {
        cost: 5,
        rewardStars: 1,
        terrainType: new Set([TerrainType.Field]),
        adjacentTypes: new Set([StructureType.Sawmill, StructureType.Windmill, StructureType.Forge]),
        limitedPerCity: true,
    },

    // Organization
    [StructureType.Farm]: {
        cost: 5,
        rewardPop: 2,
        resourceType: ResourceType.Crop,
        terrainType: new Set([TerrainType.Field]),
    },
    [StructureType.Windmill]: {
        cost: 5,
        terrainType: new Set([TerrainType.Field]),
        adjacentTypes: new Set([StructureType.Farm]),
        limitedPerCity: true,
        rewardPop: 1,
    },
    [StructureType.Embassy]: {
        cost: 5,
    },

    // Climbing
    [StructureType.Mine]: {
        cost: 5,
        rewardPop: 2,
        resourceType: ResourceType.Metal,
        terrainType: new Set([TerrainType.Mountain]),
    },
    [StructureType.Forge]: {
        rewardPop: 2,
        cost: 5,
        terrainType: new Set([TerrainType.Field]),
        adjacentTypes: new Set([StructureType.Mine]),
        limitedPerCity: true,
    },
    [StructureType.MountainTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Mountain]),
    },

    // Fishing
    [StructureType.Port]: {
        cost: 7,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Water]),
    },
    [StructureType.WaterTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Water, TerrainType.Ocean]),
    },

    // Hunting
    [StructureType.LumberHut]: {
        cost: 3,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Forest]),
    },
    [StructureType.Sawmill]: {
        cost: 5,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Field]),
        adjacentTypes: new Set([StructureType.LumberHut]),
        limitedPerCity: true,
    },
    [StructureType.ForestTemple]: {
        cost: 15,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Forest]),
    },
    
    // Special
    [StructureType.Outpost]: {
        cost: 5,
        rewardPop: 1,
        tribeType: TribeType.Polaris,
        terrainType: new Set([TerrainType.Ice]),
        // TODO, requires replacing StructureType.Port
    },
    [StructureType.IceTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: new Set([TerrainType.Ice]),
    },

    [StructureType.AltarOfPeace]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.TowerOfWisdom]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.GrandBazaar]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.EmperorsTomb]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.GateOfPower]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.ParkOfFortune]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
    [StructureType.EyeOfGod]: {
        rewardPop: 3,
        terrainType: new Set([TerrainType.Field, TerrainType.Forest, TerrainType.Water]),
    },
};

Object.freeze(StructureSettings);