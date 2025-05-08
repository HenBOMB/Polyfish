import { AbilityType, ResourceType, StructureType, TechnologyType, TribeType, UnitType } from "../types";

export type TechnologySetting =  {
    tier?: number,
    requires?: TechnologyType,
    replacesTech?: TechnologyType,
    tribeType?: TribeType,
    next?: TechnologyType[],
    unlocksResource?: ResourceType,
    unlocksStructure?: StructureType,
    unlocksAbility?: AbilityType,
    unlocksUnit?: UnitType,
    unlocksSpecialUnits?: UnitType[],
    unlocksOther?: number, // starfish, ocean navigation, etc
}

// ! Sorted by tier and clockwise per branch
export const TechnologySettings: Record<TechnologyType, TechnologySetting> = {
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
        unlocksOther: 1, // wealth
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
    [TechnologyType.Organization]: {
        tier: 1,
        next: [TechnologyType.Strategy, TechnologyType.Farming],
        unlocksResource: ResourceType.Fruit,
        // ceck nots
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
        unlocksStructure: StructureType.Embassy,
        unlocksOther: 1, // capital vision
    },
    [TechnologyType.Climbing]: {
        tier: 1,
        next: [TechnologyType.Mining, TechnologyType.Meditation],
        unlocksStructure: StructureType.MountainTemple,
        unlocksOther: 1, // pacifist
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
        unlocksOther: 1, // discount
    },
    [TechnologyType.Fishing]: {
        tier: 1,
        next: [TechnologyType.Sailing, TechnologyType.Ramming],
        unlocksResource: ResourceType.Fish,
        unlocksUnit: UnitType.Raft,
    },
    [TechnologyType.Sailing]: {
        tier: 2,
        requires: TechnologyType.Fishing,
        next: [TechnologyType.Navigation],
        unlocksUnit: UnitType.Scout,
        unlocksOther: 1, // ocean movement, also its just good
    },
    [TechnologyType.Navigation]: {
        tier: 3,
        requires: TechnologyType.Ramming,
        unlocksUnit: UnitType.Bomber,
        unlocksResource: ResourceType.Starfish,
    },
    [TechnologyType.Ramming]: {
        tier: 2,
        requires: TechnologyType.Fishing,
        next: [TechnologyType.Aquatism],
        unlocksUnit: UnitType.Rammer,
    },
    [TechnologyType.Aquatism]: {
        tier: 3,
        requires: TechnologyType.Ramming,
        unlocksStructure: StructureType.WaterTemple,
        unlocksOther: 2, // water and ocean def
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
        unlocksOther: 1, // forest def
        unlocksSpecialUnits: [UnitType.Phychi, UnitType.IceArcher],
    },
    [TechnologyType.Spiritualism]: {
        tier: 3,
        requires: TechnologyType.Archery,
        unlocksStructure: StructureType.ForestTemple,
        unlocksAbility: AbilityType.GrowForest,
    },
    [TechnologyType.Forestry]: {
        tier: 2,
        requires: TechnologyType.Hunting,
        unlocksStructure: StructureType.LumberHut,
        unlocksAbility: AbilityType.ClearForest,
        next: [TechnologyType.Mathematics],
        unlocksOther: 1,
    },
    [TechnologyType.Mathematics]: {
        tier: 3,
        requires: TechnologyType.Forestry,
        unlocksStructure: StructureType.Sawmill,
        unlocksUnit: UnitType.Catapult,
        unlocksSpecialUnits: [UnitType.Exida],
    },

    // Replaced tech

    // ! polaris
    [TechnologyType.Frostwork]: {
        replacesTech: TechnologyType.Fishing,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.Polarism],
        unlocksUnit: UnitType.Mooni,
    },
    [TechnologyType.Sledding]: {
        replacesTech: TechnologyType.Sailing,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.PolarWarfare],
        unlocksUnit: UnitType.BattleSled,
    },
    [TechnologyType.IceFishing]: {
        replacesTech: TechnologyType.Ramming,
        tribeType: TribeType.Polaris,
        next: [TechnologyType.Polarism],
    },
    [TechnologyType.PolarWarfare]: {
        replacesTech: TechnologyType.Navigation,
        tribeType: TribeType.Polaris,
        unlocksUnit: UnitType.IceFortress,
    },
    [TechnologyType.Polarism]: {
        replacesTech: TechnologyType.Aquatism,
        tribeType: TribeType.Polaris,
        unlocksStructure: StructureType.IceTemple,
    },
    // ! cymanti
    [TechnologyType.Recycling]: {
        replacesTech: TechnologyType.Construction,
        tribeType: TribeType.Cymanti,
    },
    [TechnologyType.Pascetism]: {
        replacesTech: TechnologyType.Sailing,
        tribeType: TribeType.Cymanti,
        next: [TechnologyType.Navigation],
        unlocksUnit: UnitType.Raychi,
    },
    [TechnologyType.ShockTactics]: {
        replacesTech: TechnologyType.Chivalry,
        tribeType: TribeType.Cymanti,
        unlocksUnit: UnitType.Doomux,
    },
    [TechnologyType.Hydrology]: {
        replacesTech: TechnologyType.Ramming,
        tribeType: TribeType.Cymanti,
        next: [TechnologyType.Aquatism],
    },
    // ! aquarion
    [TechnologyType.Spearing]: {
        replacesTech: TechnologyType.Chivalry,
        tribeType: TribeType.Aquarion,
    },
    [TechnologyType.Amphibian]: {
        replacesTech: TechnologyType.Riding,
        tribeType: TribeType.Aquarion,
        unlocksUnit: UnitType.Tridention,
    },
    [TechnologyType.FreeDiving]: {
        replacesTech: TechnologyType.FreeSpirit,
        tribeType: TribeType.Aquarion,
        next: [TechnologyType.Chivalry],
    },
    // ! elyrion
    [TechnologyType.ForestMagic]: {
        replacesTech: TechnologyType.Hunting,
        tribeType: TribeType.Elyrion,
        next: [TechnologyType.Archery, TechnologyType.Forestry],
    },
    [TechnologyType.Unbuildable]: {
    }
};

/**
 * ONLY all techs that are used by special units and need to be replaced by their target tier technology.
 * [TribeType] -> Returns a map of TechnologyType (source tier tech) -> TechnologyType (replaced tech)
 */
export const TechnologyReplacements: Record<TribeType, TechnologyType[]> = Object.entries(TechnologySettings)
    .filter(x => x[1].tribeType && x[1].replacesTech)
    .reduce((a: Record<TribeType, TechnologyType[]>, { 0: k, 1: v }) => ({ ...a, [v.tribeType!]: { ...a[v.tribeType!], [v.replacesTech!]: k }}), { } as any) as any;

export const TechnologyUnlockable: Record<TechnologyType, TechnologySetting> = Object.entries(TechnologySettings)
    .reduce((a, b) => (!b[1].tier || b[1].tribeType && b[1].replacesTech? a : { ...a, ...(a[b[0]] || {}), [b[0]]: b[1] }), {} as any) as any

// Sort is used by ruin rewards
export const TechnologyUnlockableList: TechnologyType[] = Object.keys(TechnologyUnlockable) as any;

Object.freeze(TechnologySettings);
Object.freeze(TechnologyReplacements);
Object.freeze(TechnologyUnlockable);
Object.freeze(TechnologyUnlockableList);