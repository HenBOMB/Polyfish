import { TribeState } from "../states";
import { TribeType, TechnologyType, UnitType, ResourceType } from "../types";

export const TribeSettings: Record<TribeType, { 
    startingTech?: TechnologyType,
    color: string,
    specialName: string,
    uniqueSuperUnit?: UnitType,
    uniqueStartingUnit?: UnitType,
    specialStart?: (tribe: TribeState) => void,
    specialTechTree?: Record<number, TechnologyType>,
}> = {
    [TribeType.None]: {
        color: "#000000",
        specialName: "None",
    },
    [TribeType.Nature]: {
        color: "#32CD32",
        specialName: "Nature",
    },
    [TribeType.XinXi]: {
        startingTech: TechnologyType.Climbing,
        color: "#cc0000",
        specialName: "Sha-po",
    },
    [TribeType.Imperius]: {
        startingTech: TechnologyType.Organization,
        color: "#0000ff",
        specialName: "Lirepacci",
    },
    [TribeType.Bardur]: {
        startingTech: TechnologyType.Hunting,
        color: "#352514",
        specialName: "Baergøff",
    },
    [TribeType.Oumaji]: {
        startingTech: TechnologyType.Riding,
        uniqueStartingUnit: UnitType.Rider,
        color: "#ffff00",
        specialName: "Khondor",
    },
    [TribeType.Kickoo]: {
        startingTech: TechnologyType.Fishing,
        color: "#00ff00",
        specialName: "Ragoo",
    },
    [TribeType.Hoodrick]: {
        startingTech: TechnologyType.Archery,
        uniqueStartingUnit: UnitType.Archer,
        color: "#996600",
        specialName: "Yorthwober",
    },
    [TribeType.Luxidoor]: {
        specialStart: (tribe: TribeState) => {
            tribe._cities[0]._level++;
        },
        color: "#ab3bd6",
        specialName: "Aumux",
    },
    [TribeType.Vengir]: {
        startingTech: TechnologyType.Smithery,
        uniqueStartingUnit: UnitType.Swordsman,
        color: "#ffffff",
        specialName: "Cultist",
    },
    [TribeType.Zebasi]: {
        startingTech: TechnologyType.Farming,
        color: "#ff9900",
        specialName: "Anzala",
    },
    [TribeType.AiMo]: {
        startingTech: TechnologyType.Meditation,
        color: "#36e2aa",
        specialName: "To-Lï",
    },
    [TribeType.Aquarion]: {
        startingTech: TechnologyType.Amphibian,
        uniqueStartingUnit: UnitType.Amphibian,
        color: "#f38381",
        specialName: "Forgotten",
        specialTechTree: {
            [TechnologyType.Chivalry]: TechnologyType.Spearing,
            [TechnologyType.FreeSpirit]: TechnologyType.FreeDiving
        }
    },
    [TribeType.Quetzali]: {
        startingTech: TechnologyType.Strategy,
        uniqueStartingUnit: UnitType.Defender,
        color: "#275c4a",
        specialName: "Iqaruz",
    },
    // TODO: elyron cant use burn forest nor clear forest
    // TODO: elydon can see ruins through clouds
    [TribeType.Elyrion]: {
        startingTech: TechnologyType.ForestMagic,
        color: "#ff0099",
        specialName: "₼idŋighţ",
        specialTechTree: {
            [TechnologyType.Hunting]: TechnologyType.ForestMagic,
        },
    },
    [TribeType.Yadakk]: {
        startingTech: TechnologyType.Roads,
        color: "#7d231c",
        specialName: "Yădakk",
    },
    [TribeType.Polaris]: {
        startingTech: TechnologyType.Frostwork,
        uniqueStartingUnit: UnitType.Mooni,
        color: "#b6a185",
        specialName: "Polaris",
        specialTechTree: {
            [TechnologyType.Fishing]: TechnologyType.Frostwork,
            [TechnologyType.Sailing]: TechnologyType.Sledding,
            [TechnologyType.Aquaculture]: TechnologyType.IceFishing,
            [TechnologyType.Navigation]: TechnologyType.PolarWarfare,
            [TechnologyType.Polarism]: TechnologyType.Aquatism,
        },
    },
    [TribeType.Cymanti]: {
        startingTech: TechnologyType.Farming,
        uniqueStartingUnit: UnitType.Shaman,
        color: "#c2fd00",
        specialName: "Cymanti",
        specialTechTree: {
            [TechnologyType.Aquaculture]: TechnologyType.Hydrology,
            [TechnologyType.Sailing]: TechnologyType.Pascetism,
            [TechnologyType.Construction]: TechnologyType.Recycling,
            [TechnologyType.Chivalry]: TechnologyType.ShockTactics,
        },
    },
};

Object.freeze(TribeSettings);