import { TribeState } from "../states";
import { TribeType, TechnologyType, UnitType, ResourceType } from "../types";

export const TribeSettings: Record<TribeType, { 
    startingTech?: TechnologyType,
    color: string,
    specialName: string,
    uniqueSuperUnit?: UnitType,
    uniqueStartingUnit?: UnitType,
    specialStart?: (tribe: TribeState) => void,
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
    [TribeType.Quetzali]: {
        startingTech: TechnologyType.Strategy,
        uniqueStartingUnit: UnitType.Defender,
        color: "#275c4a",
        specialName: "Iqaruz",
    },
    [TribeType.Yadakk]: {
        startingTech: TechnologyType.Roads,
        color: "#7d231c",
        specialName: "Yădakk",
    },
    [TribeType.Aquarion]: {
        startingTech: TechnologyType.Amphibian,
        uniqueStartingUnit: UnitType.Amphibian,
        color: "#f38381",
        specialName: "Forgotten",
        uniqueSuperUnit: UnitType.Crab,
    },
    // TODO: elyrion cant use burn forest nor clear forest
    // TODO: elyrion can see ruins through clouds
    [TribeType.Elyrion]: {
        startingTech: TechnologyType.ForestMagic,
        color: "#ff0099",
        specialName: "₼idŋighţ",
        uniqueSuperUnit: UnitType.DragonEgg,
    },
    [TribeType.Polaris]: {
        startingTech: TechnologyType.Frostwork,
        uniqueStartingUnit: UnitType.Mooni,
        color: "#b6a185",
        specialName: "Polaris",
        uniqueSuperUnit: UnitType.Gaami,
    },
    [TribeType.Cymanti]: {
        startingTech: TechnologyType.Farming,
        uniqueStartingUnit: UnitType.Shaman,
        color: "#c2fd00",
        specialName: "Cymanti",
        uniqueSuperUnit: UnitType.Centipede,
    },
};

export const TribeTypeCount = Object.keys(TribeSettings).length - 2;

Object.freeze(TribeSettings);
Object.freeze(TribeTypeCount);