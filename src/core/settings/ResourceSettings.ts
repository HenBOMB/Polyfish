import { TechnologyType, StructureType, ResourceType, TribeType } from "../types";

export const ResourceSettings: Record<ResourceType, {
    cost?: number;
    techRequired: TechnologyType,
    structType?: StructureType,
    visibleRequired?: TechnologyType[],
    requiresCapture?: true,
    rewardPop: number;
    rewardStars?: number;
    tribeType?: TribeType;
}> = {
    [ResourceType.None]: {
        techRequired: TechnologyType.Unrequired,
        rewardPop: 0,
    },
    [ResourceType.Game]: {
        cost: 2,
        techRequired: TechnologyType.Hunting,
        rewardPop: 1,
    },
    [ResourceType.Crop]: {
        techRequired: TechnologyType.Farming,
        structType: StructureType.Farm,
        visibleRequired: [TechnologyType.Organization, TechnologyType.Farming, TechnologyType.Construction],
        rewardPop: 2,
    },
    [ResourceType.Fish]: {
        cost: 2,
        techRequired: TechnologyType.Fishing,
        rewardPop: 1,
    },
    [ResourceType.Metal]: {
        techRequired: TechnologyType.Mining,
        structType: StructureType.Mine,
        visibleRequired: [TechnologyType.Climbing, TechnologyType.Mining, TechnologyType.Smithery],
        rewardPop: 2,
    },
    [ResourceType.Fruit]: {
        cost: 2,
        techRequired: TechnologyType.Organization,
        rewardPop: 1
    },
    [ResourceType.Spores]: {
        techRequired: TechnologyType.Unrequired,
        structType: StructureType.Spores,
        rewardPop: 1,
        tribeType: TribeType.Cymanti,
    },
    [ResourceType.Starfish]: {
        techRequired: TechnologyType.Navigation,
        visibleRequired: [TechnologyType.Fishing, TechnologyType.Sailing, TechnologyType.Navigation],
        requiresCapture: true,
        rewardStars: 5,
        rewardPop: 0,
    },
    [ResourceType.AquaCrop]: {
        techRequired: TechnologyType.BeyondComprehension,
        rewardPop: 2,
    },
};

// ! freeze everything
Object.freeze(ResourceSettings);