import { TileState, TribeState } from "../states";
import { TerrainType, ResourceType, StructureType, TribeType } from "../types";

// ! Sorted by tier and clockwise per branch
export const StructureSettings: Record<StructureType, { 
    cost?: number, 
    terrainType?: TerrainType[], 
    adjacentTypes?: StructureType[], 
    resourceType?: ResourceType, 
    limitedPerCity?: true,
    task?: (tribe: TribeState, tiles: Record<number, TileState>) => boolean,
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
        terrainType: [TerrainType.Forest],
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Swamp]: {
        // TODO Spores
        // techType: TechnologyType.None,
        terrainType: [TerrainType.Ocean],
    },
    // TODO Mycelium
    [StructureType.Mycelium]: {
        // techType: TechnologyType.None,
        terrainType: [TerrainType.Field],
        limitedPerCity: true,
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Lighthouse]: {
    },

    // Riding
    [StructureType.Road]: {
        cost: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest], 
    },
    [StructureType.Bridge]: {
        terrainType: [TerrainType.Water],
        // techType: TechnologyType.Roads,
        // terrainType: TerrainType.Water,
        // TODO Bridge logic for connected adjacent ground tiles or whatever
    },
    [StructureType.Temple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: [TerrainType.Field],
    },
    [StructureType.Market]: {
        cost: 5,
        rewardStars: 1,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Sawmill, StructureType.Windmill, StructureType.Forge],
        limitedPerCity: true,
    },

    // Organization
    [StructureType.Farm]: {
        cost: 5,
        rewardPop: 2,
        resourceType: ResourceType.Crop,
        terrainType: [TerrainType.Field],
    },
    [StructureType.Windmill]: {
        cost: 5,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Farm],
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
        terrainType: [TerrainType.Mountain],
    },
    [StructureType.Forge]: {
        rewardPop: 2,
        cost: 5,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Mine],
        limitedPerCity: true,
    },
    [StructureType.MountainTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: [TerrainType.Mountain],
    },

    // Fishing
    [StructureType.Port]: {
        cost: 7,
        rewardPop: 1,
        terrainType: [TerrainType.Water],
    },
    [StructureType.WaterTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: [TerrainType.Water, TerrainType.Ocean],
    },

    // Hunting
    [StructureType.LumberHut]: {
        cost: 3,
        rewardPop: 1,
        terrainType: [TerrainType.Forest],
    },
    [StructureType.Sawmill]: {
        cost: 5,
        rewardPop: 1,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.LumberHut],
        limitedPerCity: true,
    },
    [StructureType.ForestTemple]: {
        cost: 15,
        rewardPop: 1,
        terrainType: [TerrainType.Forest],
    },
    
    // Special
    [StructureType.Outpost]: {
        cost: 5,
        rewardPop: 1,
        tribeType: TribeType.Polaris,
        terrainType: [TerrainType.Ice],
        // TODO, requires replacing StructureType.Port
    },
    [StructureType.IceTemple]: {
        cost: 20,
        rewardPop: 1,
        terrainType: [TerrainType.Ice],
    },

    [StructureType.AltarOfPeace]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
    },
    [StructureType.TowerOfWisdom]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
    },
    [StructureType.GrandBazaar]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._cities.reduce((acc, cur) => acc + (cur._connectedToCapital? 1 : 0), 0) > 4;
        }
    },
    [StructureType.EmperorsTomb]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._stars > 99;
        }
    },
    [StructureType.GateOfPower]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._kills > 9;
        }
    },
    [StructureType.ParkOfFortune]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        // ! level up, any level 5 city
        task: (tribe: TribeState, _: any) => { 
            return tribe._cities.some(city => city._level > 4);
        }
    },
    [StructureType.EyeOfGod]: {
        rewardPop: 3,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        // ! navigation, all tiles explored
        task: (tribe: TribeState, tiles: Record<number, TileState>) => {
            return Object.values(tiles).every(x => x._explorers.includes(tribe.owner));
        }
    },
};

Object.freeze(StructureSettings);