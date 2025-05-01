import { AbilityType, ResourceType, StructureType, TechnologyType, TribeType, UnitType } from "../types";

export const TechnologySettings: Record<TechnologyType, {
    tier?: number,
    requires?: TechnologyType,
    replaced?: TechnologyType,
    tribeType?: TribeType,
    next?: TechnologyType[],
    unlocksResource?: ResourceType,
    unlocksStructure?: StructureType,
    unlocksAbility?: AbilityType,
    unlocksUnit?: UnitType,
    unlocksSpecialUnits?: UnitType[],
}> = {
    [TechnologyType.None]: {
        tier: 0,
        unlocksUnit: UnitType.Warrior,
    },
    [TechnologyType.Riding]: {
        tier: 1,
        next: [TechnologyType.FreeSpirit, TechnologyType.Roads],
        unlocksUnit: UnitType.Rider,
        unlocksSpecialUnits: [UnitType.Hexapod, UnitType.Amphibian],
    },
    [TechnologyType.FreeSpirit]: {
        tier: 2,
        requires: TechnologyType.Riding,
        unlocksStructure: StructureType.Temple,
        unlocksAbility: AbilityType.Disband,
        next: [TechnologyType.Chivalry],
    },
    [TechnologyType.Chivalry]: {
        tier: 3,
        requires: TechnologyType.FreeSpirit,
        unlocksUnit: UnitType.Knight,
        unlocksSpecialUnits: [UnitType.Tridention],
        unlocksAbility: AbilityType.Destroy
    },
    [TechnologyType.Roads]: {
        tier: 2,
        requires: TechnologyType.Riding,
        next: [TechnologyType.Trade],
        unlocksStructure: StructureType.Bridge,
    },
    [TechnologyType.Trade]: {
        tier: 3,
        requires: TechnologyType.Roads,
        unlocksStructure: StructureType.Market,
    },
    [TechnologyType.Organization]: {
        tier: 1,
        next: [TechnologyType.Strategy, TechnologyType.Farming],
        unlocksResource: ResourceType.Fruit,
    },
    [TechnologyType.Farming]: {
        tier: 2,
        requires: TechnologyType.Organization,
        next: [TechnologyType.Construction],
        unlocksResource: ResourceType.Crop,
        unlocksStructure: StructureType.Farm,
    },
    [TechnologyType.Construction]: {
        tier: 3,
        requires: TechnologyType.Farming,
        unlocksStructure: StructureType.Windmill,
        unlocksAbility: AbilityType.BurnForest
    },
    [TechnologyType.Strategy]: {
        tier: 2,
        requires: TechnologyType.Organization,
        unlocksUnit: UnitType.Defender,
        unlocksSpecialUnits: [UnitType.Kiton],
    },
    [TechnologyType.Diplomacy]: {
        tier: 3,
        requires: TechnologyType.Strategy,
        unlocksUnit: UnitType.Cloak,
    },
    [TechnologyType.Climbing]: {
        tier: 1,
        next: [TechnologyType.Mining, TechnologyType.Meditation],
    },
    [TechnologyType.Mining]: {
        tier: 2,
        requires: TechnologyType.Climbing,
        next: [TechnologyType.Smithery],
        unlocksResource: ResourceType.Metal,
        unlocksStructure: StructureType.Mine,
    },
    [TechnologyType.Smithery]: {
        tier: 3,
        requires: TechnologyType.Mining,
        unlocksStructure: StructureType.Forge,
        unlocksUnit: UnitType.Swordsman,
    },
    [TechnologyType.Meditation]: {
        tier: 2,
        requires: TechnologyType.Climbing,
        next: [TechnologyType.Philosophy],
        unlocksStructure: StructureType.MountainTemple,
    },
    [TechnologyType.Philosophy]: {
        tier: 3,
        requires: TechnologyType.Meditation,
        unlocksUnit: UnitType.MindBender,
        unlocksSpecialUnits: [UnitType.Shaman],
    },
    [TechnologyType.Fishing]: {
        tier: 1,
        next: [TechnologyType.Sailing, TechnologyType.Aquaculture],
        unlocksResource: ResourceType.Fish,
        unlocksUnit: UnitType.Raft,
    },
    [TechnologyType.Sailing]: {
        tier: 2,
        requires: TechnologyType.Fishing,
        next: [TechnologyType.Navigation],
        unlocksResource: ResourceType.Starfish,
        unlocksUnit: UnitType.Scout,
    },
    [TechnologyType.Navigation]: {
        tier: 3,
        requires: TechnologyType.Aquaculture,
        unlocksUnit: UnitType.Bomber,
        unlocksAbility: AbilityType.StarfishHarvesting
    },
    [TechnologyType.Aquaculture]: {
        tier: 2,
        requires: TechnologyType.Fishing,
        next: [TechnologyType.Aquatism],
        unlocksUnit: UnitType.Rammer,
    },
    [TechnologyType.Aquatism]: {
        tier: 3,
        requires: TechnologyType.Aquaculture,
        unlocksUnit: UnitType.Rammer,
        unlocksStructure: StructureType.WaterTemple,
    },
    [TechnologyType.Hunting]: {
        tier: 1,
        next: [TechnologyType.Archery, TechnologyType.Forestry],
        unlocksResource: ResourceType.WildAnimal,
    },
    [TechnologyType.Archery]: {
        tier: 2,
        requires: TechnologyType.Hunting,
        next: [TechnologyType.Spiritualism],
        unlocksUnit: UnitType.Archer,
        unlocksSpecialUnits: [UnitType.Phychi, UnitType.IceArcher],
    },
    [TechnologyType.Spiritualism]: {
        tier: 3,
        requires: TechnologyType.Archery,
        unlocksStructure: StructureType.ForestTemple,
        unlocksAbility: AbilityType.GrowForest
    },
    [TechnologyType.Forestry]: {
        tier: 2,
        requires: TechnologyType.Hunting,
        unlocksStructure: StructureType.LumberHut,
        unlocksAbility: AbilityType.ClearForest,
        next: [TechnologyType.Mathematics],
    },
    [TechnologyType.Mathematics]: {
        tier: 3,
        requires: TechnologyType.Forestry,
        unlocksStructure: StructureType.Sawmill,
        unlocksUnit: UnitType.Catapult,
        unlocksSpecialUnits: [UnitType.Exida],
    },
    // ! polaris
    [TechnologyType.Frostwork]: {
        replaced: TechnologyType.Fishing,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.Polarism],
        unlocksUnit: UnitType.Mooni,
    },
    [TechnologyType.Sledding]: {
        replaced: TechnologyType.Sailing,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.PolarWarfare],
        unlocksUnit: UnitType.BattleSled,
    },
    [TechnologyType.IceFishing]: {
        replaced: TechnologyType.Aquaculture,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.Polarism],
    },
    [TechnologyType.PolarWarfare]: {
        replaced: TechnologyType.Navigation,
        tribeType: TribeType.Polaris,
        unlocksUnit: UnitType.IceFortress,
    },
    [TechnologyType.Polarism]: {
        replaced: TechnologyType.Aquatism,
        tribeType: TribeType.Polaris,
        unlocksStructure: StructureType.IceTemple,
    },
    // ! cymanti
    [TechnologyType.Recycling]: {
        replaced: TechnologyType.Construction,
        tribeType: TribeType.Cymanti,
    },
    [TechnologyType.Pascetism]: {
        replaced: TechnologyType.Sailing,
        tribeType: TribeType.Cymanti,
        next: [TechnologyType.Navigation],
        unlocksUnit: UnitType.Raychi,
    },
    [TechnologyType.ShockTactics]: {
        replaced: TechnologyType.Chivalry,
        tribeType: TribeType.Cymanti,
        unlocksUnit: UnitType.Doomux,
    },
    [TechnologyType.Hydrology]: {
        replaced: TechnologyType.Aquaculture,
        tribeType: TribeType.Cymanti,
        next: [TechnologyType.Aquatism],
    },
    // ! aquarion
    [TechnologyType.Spearing]: {
        replaced: TechnologyType.Chivalry,
        tribeType: TribeType.Aquarion,
    },
    [TechnologyType.Amphibian]: {
        replaced: TechnologyType.Riding,
        tribeType: TribeType.Aquarion,
    },
    [TechnologyType.FreeDiving]: {
        replaced: TechnologyType.FreeSpirit,
        tribeType: TribeType.Aquarion,
        next: [TechnologyType.Chivalry],
    },
    // ! elyrion
    [TechnologyType.ForestMagic]: {
        replaced: TechnologyType.Hunting,
        tribeType: TribeType.Elyrion,
        next: [TechnologyType.Archery, TechnologyType.Forestry],
    },
    // ! TODO ??
    // [TechnologyType.CymantiFishing]: {
    //     // tier: 2,
    //     // replaced: TechnologyType.Fishing,
    // },
    [TechnologyType.Unbuildable]: {
    }
};

Object.freeze(TechnologySettings);