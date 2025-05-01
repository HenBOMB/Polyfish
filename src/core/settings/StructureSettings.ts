import { TileState, TribeState } from "../states";
import { TechnologyType, TerrainType, ResourceType, StructureType, TribeType } from "../types";

export const StructureSettings: Record<StructureType, { 
    techRequired: TechnologyType,
    cost?: number, 
    terrainType: TerrainType, 
    adjacentTypes?: StructureType[], 
    resourceType?: ResourceType, 
    limitedPerCity?: true,
    task?: (tribe: TribeState, tiles: Record<number, TileState>) => boolean,
    rewardPop?: number;
    rewardStars?: number;
    tribeType?: TribeType;
}> = {
    [StructureType.None]:           {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.None,
    },
    [StructureType.Village]: {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.Field,
    },
    [StructureType.Ruin]: {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.None,
    },
    [StructureType.Farm]: {
        cost: 5,
        rewardPop: 2,
        techRequired: TechnologyType.Farming,
        resourceType: ResourceType.Crop,
        terrainType: TerrainType.Field,
    },
    [StructureType.Windmill]: {
        cost: 5,
        techRequired: TechnologyType.Construction,
        terrainType: TerrainType.Field,
        adjacentTypes: [StructureType.Farm],
        limitedPerCity: true,
        rewardPop: 1,
    },
    [StructureType.Port]: {
        cost: 7,
        rewardPop: 1,
        techRequired: TechnologyType.Fishing,
        terrainType: TerrainType.Water,
    },
    [StructureType.LumberHut]: {
        cost: 3,
        rewardPop: 1,
        techRequired: TechnologyType.Forestry,
        terrainType: TerrainType.Forest,
    },
    [StructureType.Sawmill]: {
        cost: 5,
        rewardPop: 1,
        techRequired: TechnologyType.Mathematics,
        terrainType: TerrainType.Field,
        adjacentTypes: [StructureType.LumberHut],
        limitedPerCity: true,
    },
    [StructureType.Temple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.FreeSpirit,
        terrainType: TerrainType.Field,
    },
    [StructureType.ForestTemple]: {
        cost: 15,
        rewardPop: 1,
        techRequired: TechnologyType.Spiritualism,
        terrainType: TerrainType.Forest,
    },
    [StructureType.WaterTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Aquatism,
        terrainType: TerrainType.Ocean,
    },
    [StructureType.MountainTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Meditation,
        terrainType: TerrainType.Mountain,
    },
    [StructureType.IceTemple]: {
        cost: 20,
        rewardPop: 1,
        techRequired: TechnologyType.Polarism,
        terrainType: TerrainType.Ice,
    },
    [StructureType.Mine]: {
        cost: 5,
        rewardPop: 2,
        techRequired: TechnologyType.Mining,
        resourceType: ResourceType.Metal,
        terrainType: TerrainType.Mountain,
    },
    [StructureType.Forge]: {
        rewardPop: 2,
        cost: 5,
        techRequired: TechnologyType.Smithery,
        terrainType: TerrainType.Field,
        adjacentTypes: [StructureType.Mine],
        limitedPerCity: true,
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
        terrainType: TerrainType.GroundWater,
    },
    [StructureType.TowerOfWisdom]: {
        rewardPop: 3,
        techRequired: TechnologyType.Philosophy,
        // ! philosophy, research all technologies
        terrainType: TerrainType.GroundWater,
        task: (tribe: TribeState, _: any) => {
            return tribe._tech.length >= Object.values(TechnologyType).length - 2; // none and unbuildable
        }
    },
    [StructureType.GrandBazaar]: {
        rewardPop: 3,
        techRequired: TechnologyType.Roads,
        terrainType: TerrainType.GroundWater,
        // ! roads, 5 cities connected to capital
        task: (tribe: TribeState, _: any) => {
            return tribe._cities.reduce((acc, cur) => acc + (cur._connectedToCapital? 1 : 0), 0) > 4;
        }
    },
    [StructureType.EmperorsTomb]:   {
        rewardPop: 3,
        techRequired: TechnologyType.Trade,
        terrainType: TerrainType.GroundWater,
        // ! trade, 100 stars
        task: (tribe: TribeState, _: any) => {
            return tribe._stars > 99;
        }
    },
    [StructureType.GateOfPower]:    {
        rewardPop: 3,
        techRequired: TechnologyType.None,
        terrainType: TerrainType.GroundWater,
        // ! kill, 10 kills
        task: (tribe: TribeState, _: any) => {
            return tribe._kills > 9;
        }
    },
    [StructureType.ParkOfFortune]:  {
        rewardPop: 3,
        techRequired: TechnologyType.None,
        terrainType: TerrainType.GroundWater,
        // ! level up, any level 5 city
        task: (tribe: TribeState, _: any) => { 
            return tribe._cities.some(city => city._level > 4);
        }
    },
    [StructureType.EyeOfGod]:       {
        rewardPop: 3,
        techRequired: TechnologyType.Navigation,
        terrainType: TerrainType.GroundWater,
        // ! navigation, all tiles explored
        task: (tribe: TribeState, tiles: Record<number, TileState>) => {
            return Object.values(tiles).every(x => x.explorers.includes(tribe.owner));
        }
    },
    [StructureType.IcePort]:        {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.Ice,
        tribeType: TribeType.Polaris,
        // TODO IcePort
        // techType: TechnologyType.Fishing,
        // terrainType: TerrainType.Ice,
    },
    [StructureType.Spores]:         {
        techRequired: TechnologyType.Farming,
        resourceType: ResourceType.Spores,
        terrainType: TerrainType.Forest,
        tribeType: TribeType.Cymanti,
    },
    [StructureType.Swamp]:          {
        // TODO Spores
        techRequired: TechnologyType.Unbuildable,
        // techType: TechnologyType.None,
        terrainType: TerrainType.Ocean,
    },
    [StructureType.Mycelium]:       {
        techRequired: TechnologyType.Unbuildable,
        // techType: TechnologyType.None,
        terrainType: TerrainType.Field,
        limitedPerCity: true,
        tribeType: TribeType.Cymanti,
        // TODO Mycelium
    },
    [StructureType.Lighthouse]:     {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.None,
    },
    [StructureType.Bridge]:         {
        techRequired: TechnologyType.Unbuildable,
        terrainType: TerrainType.Water,
        // techType: TechnologyType.Roads,
        // terrainType: TerrainType.Water,
        // TODO Bridge logic for connected adjacent ground tiles or whatever
    },
    [StructureType.Market]:         {
        cost: 5,
        rewardStars: 1,
        techRequired: TechnologyType.Trade,
        terrainType: TerrainType.Field,
        adjacentTypes: [StructureType.Sawmill, StructureType.Windmill, StructureType.Forge],
        limitedPerCity: true,
    },
    // [StructureType.Road]:           {
    //     cost: 3,
    //     techRequired: TechnologyType.Roads,
    //     TODO any terrain type except mountains, water, ice, and Algae
    //     terrainType: TerrainType.Field, 
    //     adjacentTypes: [StructureType.Sawmill, StructureType.Windmill, StructureType.Forge],
    //     limitedPerCity: true,
    // },
};

// ! pre calculate terrain types for all structures, quick lookup
export const StructureByTerrain: Record<TerrainType, StructureType[]> = Object.keys(StructureSettings).reduce((acc: Record<TerrainType, StructureType[]>, numId: unknown) => {
    const structType = numId as StructureType;
    const setting = StructureSettings[structType];
    
    if(!setting) {
        console.warn(`No settings found for structure "${structType}", skipping..`);
        return acc;
    }

    // ! skip invalid, none, unbuildable, unknown terrain
    if(setting.techRequired == TechnologyType.Unbuildable || !setting.terrainType) return acc;
    
    const terrainType = setting.terrainType! as TerrainType;
    
    acc[terrainType].push(structType);
    
    switch (terrainType) {
        // ! ocean can also be placed on water
        case TerrainType.Ocean:
        acc[TerrainType.Water].push(structType);
        break;
        
        // ! land can also be placed on forest
        case TerrainType.Field:
        acc[TerrainType.Forest].push(structType);
        break;
        
        // ! unique to task structures, can be placed on water, but not ocean and land but not forest
        case TerrainType.GroundWater:
        acc[TerrainType.Field].push(structType);
        acc[TerrainType.Water].push(structType);
        break;
        
        // ! i think ice can be placed on water as well
        case TerrainType.Ice:
        acc[TerrainType.Water].push(structType);
        break;
        
        default:
        break;
    }
    
    return acc;
}, Object.values(TerrainType).filter(x => typeof x == 'number').reduce((acc, cur) => ({ ...acc, [cur as unknown as TerrainType]: [] }), {}) as Record<TerrainType, StructureType[]>);

Object.freeze(StructureSettings);
Object.freeze(StructureByTerrain);