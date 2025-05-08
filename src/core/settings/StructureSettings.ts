import { TileState, TribeState } from "../states";
import { TechnologyType, TerrainType, ResourceType, StructureType, TribeType } from "../types";
import { TechnologyUnlockable } from "./TechnologySettings";

// ! Sorted by tier and clockwise per branch
export const StructureSettings: Record<StructureType, { 
    techRequired: TechnologyType,
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
        techRequired: TechnologyType.Unbuildable,
    },
    [StructureType.Village]: {
        techRequired: TechnologyType.Unbuildable,
        terrainType: [TerrainType.Field],
    },
    [StructureType.Ruin]: {
        techRequired: TechnologyType.Unbuildable,
    },
    // TODO not sure how this works but, i think harvesting spores leads to swamp? or sumthin
    [StructureType.Spores]: {
        techRequired: TechnologyType.Unbuildable,
        resourceType: ResourceType.Spores,
        terrainType: [TerrainType.Forest],
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Swamp]: {
        // TODO Spores
        techRequired: TechnologyType.Unbuildable,
        // techType: TechnologyType.None,
        terrainType: [TerrainType.Ocean],
    },
    // TODO Mycelium
    [StructureType.Mycelium]: {
        techRequired: TechnologyType.Unbuildable,
        // techType: TechnologyType.None,
        terrainType: [TerrainType.Field],
        limitedPerCity: true,
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Lighthouse]: {
        techRequired: TechnologyType.Unbuildable,
    },

    // Riding
    [StructureType.Road]: {
        cost: 3,
        techRequired: TechnologyType.Roads,
        terrainType: [TerrainType.Field, TerrainType.Forest], 
    },
    [StructureType.Bridge]: {
        techRequired: TechnologyType.Unbuildable,
        terrainType: [TerrainType.Water],
        // techType: TechnologyType.Roads,
        // terrainType: TerrainType.Water,
        // TODO Bridge logic for connected adjacent ground tiles or whatever
    },
    [StructureType.Temple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.FreeSpirit,
        terrainType: [TerrainType.Field],
    },
    [StructureType.Market]: {
        cost: 5,
        rewardStars: 1,
        techRequired: TechnologyType.Trade,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Sawmill, StructureType.Windmill, StructureType.Forge],
        limitedPerCity: true,
    },

    // Organization
    [StructureType.Farm]: {
        cost: 5,
        rewardPop: 2,
        techRequired: TechnologyType.Farming,
        resourceType: ResourceType.Crop,
        terrainType: [TerrainType.Field],
    },
    [StructureType.Windmill]: {
        cost: 5,
        techRequired: TechnologyType.Construction,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Farm],
        limitedPerCity: true,
        rewardPop: 1,
    },
    [StructureType.Embassy]: {
        cost: 5,
        techRequired: TechnologyType.Unbuildable
    },

    // Climbing
    [StructureType.Mine]: {
        cost: 5,
        rewardPop: 2,
        techRequired: TechnologyType.Mining,
        resourceType: ResourceType.Metal,
        terrainType: [TerrainType.Mountain],
    },
    [StructureType.Forge]: {
        rewardPop: 2,
        cost: 5,
        techRequired: TechnologyType.Smithery,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.Mine],
        limitedPerCity: true,
    },
    [StructureType.MountainTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Meditation,
        terrainType: [TerrainType.Mountain],
    },

    // Fishing
    [StructureType.Port]: {
        cost: 7,
        rewardPop: 1,
        techRequired: TechnologyType.Fishing,
        terrainType: [TerrainType.Water],
    },
    [StructureType.WaterTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Aquatism,
        terrainType: [TerrainType.Water, TerrainType.Ocean],
    },

    // Hunting
    [StructureType.LumberHut]: {
        cost: 3,
        rewardPop: 1,
        techRequired: TechnologyType.Forestry,
        terrainType: [TerrainType.Forest],
    },
    [StructureType.Sawmill]: {
        cost: 5,
        rewardPop: 1,
        techRequired: TechnologyType.Mathematics,
        terrainType: [TerrainType.Field],
        adjacentTypes: [StructureType.LumberHut],
        limitedPerCity: true,
    },
    [StructureType.ForestTemple]: {
        cost: 15,
        rewardPop: 1,
        techRequired: TechnologyType.Spiritualism,
        terrainType: [TerrainType.Forest],
    },
    
    // Special
    [StructureType.Outpost]: {
        cost: 5,
        rewardPop: 1,
        techRequired: TechnologyType.Unbuildable,
        tribeType: TribeType.Polaris,
        terrainType: [TerrainType.Ice],
        // TODO, requires replacing StructureType.Port
    },
    [StructureType.IceTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Polarism,
        terrainType: [TerrainType.Ice],
    },

    [StructureType.AltarOfPeace]: {
        rewardPop: 3,
        techRequired: TechnologyType.Unbuildable, // TODO
        // techId: TechnologyType.Meditation,
        // ! meditation, dont kill for 5 turns
        // [StructureType.AltarOfPeace]: (state: GameState, tribe: TribeState) => {
        // 	if(tribe.builtUniqueStructures[StructureType.AltarOfPeace]) return false;
        //  tribe.tasks.some(x => x.customData )
        // 	return hasTech(tribe, TechnologyType.Meditation); 
        // },
        // ! no way to track streak (maybe customData from task?)
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
    },
    [StructureType.TowerOfWisdom]: {
        rewardPop: 3,
        techRequired: TechnologyType.Philosophy,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._tech.length == Object.values(TechnologyUnlockable).length;
        }
    },
    [StructureType.GrandBazaar]: {
        rewardPop: 3,
        techRequired: TechnologyType.Roads,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._cities.reduce((acc, cur) => acc + (cur._connectedToCapital? 1 : 0), 0) > 4;
        }
    },
    [StructureType.EmperorsTomb]: {
        rewardPop: 3,
        techRequired: TechnologyType.Trade,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._stars > 99;
        }
    },
    [StructureType.GateOfPower]: {
        rewardPop: 3,
        techRequired: TechnologyType.None,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        task: (tribe: TribeState, _: any) => {
            return tribe._kills > 9;
        }
    },
    [StructureType.ParkOfFortune]: {
        rewardPop: 3,
        techRequired: TechnologyType.None,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        // ! level up, any level 5 city
        task: (tribe: TribeState, _: any) => { 
            return tribe._cities.some(city => city._level > 4);
        }
    },
    [StructureType.EyeOfGod]: {
        rewardPop: 3,
        techRequired: TechnologyType.Navigation,
        terrainType: [TerrainType.Field, TerrainType.Forest, TerrainType.Water],
        // ! navigation, all tiles explored
        task: (tribe: TribeState, tiles: Record<number, TileState>) => {
            return Object.values(tiles).every(x => x._explorers.includes(tribe.owner));
        }
    },
};

// ! pre calculate terrain types for all structures, quick lookup
export const StructureByTerrain: Record<TerrainType, StructureType[]> = Object.keys(StructureSettings).reduce((acc: Record<TerrainType, StructureType[]>, numId: unknown) => {
    const structType = numId as StructureType;
    const settings = StructureSettings[structType];

    if(settings.techRequired == TechnologyType.Unbuildable) return acc;

    if(settings.terrainType) {
        for(const terrainType of settings.terrainType) {
            acc[terrainType].push(structType);
        }
    }
    
    return acc;
}, Object.values(TerrainType).filter(x => typeof x == 'number').reduce((acc, cur) => ({ ...acc, [cur as unknown as TerrainType]: [] }), {}) as Record<TerrainType, StructureType[]>);

Object.freeze(StructureSettings);
Object.freeze(StructureByTerrain);