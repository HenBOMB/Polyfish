import { SkillType, TechnologyType, TribeType, UnitType } from "../types";

export const UnitSettings: Record<UnitType, { 
    techType: TechnologyType,
    cost: number, 
    attack: number, 
    movement: number, 
    defense: number, 
    range: number, 
    health?: number, 
    upgradeFrom?: UnitType,
    veteran?: boolean | true,
    explodeOnly?: boolean | true,
    skills: SkillType[],
    becomes?: UnitType;
    tribeType?: TribeType;
}> = {
    [UnitType.None]: {
        techType: TechnologyType.Unbuildable,
        cost: -1,
        attack: 0,
        movement: 0,
        defense: 0,
        range: 0,
        health: 1,
        skills: []
    },
    [UnitType.Warrior]: {
        techType: TechnologyType.None,
        cost: 2,
        attack: 2,
        movement: 1,
        defense: 2,
        range: 1,
        health: 10,
        skills: [SkillType.Dash, SkillType.Fortify]
    },
    [UnitType.Rider]: {
        techType: TechnologyType.Riding,
        cost: 3,
        attack: 2,
        movement: 2,
        defense: 1,
        range: 1,
        health: 10,
        skills: [SkillType.Dash, SkillType.Escape, SkillType.Fortify]
    },
    [UnitType.Knight]: {
        techType: TechnologyType.Chivalry,
        cost: 8,
        attack: 3.5,
        movement: 3,
        defense: 1,
        range: 1,
        health: 10,
        skills: [SkillType.Dash, SkillType.Persist, SkillType.Fortify]
    },
    [UnitType.Defender]: {
        techType: TechnologyType.Strategy,
        cost: 3,
        attack: 1,
        movement: 1,
        defense: 3,
        range: 1,
        health: 15,
        skills: [SkillType.Fortify]
    },
    [UnitType.Catapult]: {
        techType: TechnologyType.Mathematics,
        cost: 8,
        attack: 4,
        movement: 1,
        defense: 0,
        range: 3,
        health: 10,
        skills: [SkillType.Stiff]
    },
    [UnitType.Archer]: {
        techType: TechnologyType.Archery,
        cost: 3,
        health: 10,
        attack: 2,
        movement: 1,
        defense: 1,
        range: 2,
        skills: [SkillType.Dash, SkillType.Fortify]
    },
    [UnitType.MindBender]: {
        techType: TechnologyType.Philosophy,
        cost: 5,
        health: 10,
        attack: 0,
        movement: 1,
        defense: 1,
        range: 1,
        skills: [SkillType.Heal, SkillType.Convert]
    },
    [UnitType.Swordsman]: {
        techType: TechnologyType.Smithery,
        cost: 5,
        health: 15,
        attack: 3,
        movement: 1,
        defense: 3,
        range: 1,
        skills: [SkillType.Dash]
    },
    [UnitType.Giant]: {
        techType: TechnologyType.Unbuildable,
        cost: 10,
        health: 40,
        attack: 5,
        movement: 1,
        defense: 4,
        range: 1,
        skills: []
    },    
    [UnitType.Cloak]: {
        techType: TechnologyType.Diplomacy,
        cost: 8,
        health: 5,
        attack: 0,
        movement: 2,
        defense: 0.5,
        range: 1,
        veteran: false,
        skills: [SkillType.Hide, SkillType.Infiltrate, SkillType.Dash, SkillType.Scout, SkillType.Creep]
    },
    [UnitType.Dagger]: {
        techType: TechnologyType.Diplomacy,
        cost: 5, // TODO is this accurate?
        health: 5,
        attack: 2,
        movement: 1,
        defense: 2,
        range: 1,
        veteran: false,
        skills: [SkillType.Dash, SkillType.Surprise, SkillType.Independent]
    },
    [UnitType.Pirate]: {
        techType: TechnologyType.Sailing,
        cost: 5, // TODO is this accurate?
        attack: 2,
        movement: 2,
        defense: 2,
        range: 1,
        veteran: false,
        skills: [SkillType.Carry, SkillType.Float, SkillType.Dash, SkillType.Surprise, SkillType.Independent]
    },
    [UnitType.Dinghy]: {
        techType: TechnologyType.Diplomacy,
        cost: 8,
        health: 5,
        attack: 0,
        movement: 2,
        defense: 0.5,
        range: 1,
        veteran: false,
        skills: [SkillType.Carry, SkillType.Float, SkillType.Hide, SkillType.Infiltrate, SkillType.Dash, SkillType.Scout]
    },

    // Navy
    [UnitType.Raft]: {
        techType: TechnologyType.Sailing,
        cost: 0,
        attack: 0,
        movement: 2,
        defense: 2,
        range: 2,
        veteran: false,
        skills: [SkillType.Carry, SkillType.Float]
    },
    [UnitType.Scout]: {
        techType: TechnologyType.Sailing,
        cost: 5,
        attack: 2,
        movement: 3,
        defense: 2,
        range: 2,
        veteran: false,
        upgradeFrom: UnitType.Raft,
        skills: [SkillType.Carry, SkillType.Dash, SkillType.Float]
    },
    [UnitType.Rammer]: {
        techType: TechnologyType.Aquatism,
        upgradeFrom: UnitType.Raft,
        cost: 5,
        attack: 3,
        movement: 3,
        defense: 3,
        range: 1,
        skills: [SkillType.Dash, SkillType.Float, SkillType.Carry]
    },
    [UnitType.Bomber]: {
        techType: TechnologyType.Navigation,
        cost: 5,
        attack: 3,
        movement: 2,
        defense: 2,
        range: 3,
        upgradeFrom: UnitType.Raft,
        skills: [SkillType.Carry, SkillType.Float, SkillType.Splash, SkillType.Stiff]
    },
    [UnitType.Juggernaut]: {
        techType: TechnologyType.Unbuildable,
        cost: 10,
        health: 40,
        attack: 4,
        movement: 2,
        defense: 4,
        range: 1,
        veteran: false,
        skills: [SkillType.Float, SkillType.Carry, SkillType.Stiff, SkillType.Stomp]
    },

    // Aquarion
    [UnitType.Crab]: {
        tribeType: TribeType.Aquarion,
        techType: TechnologyType.Unbuildable,
        cost: 10,
        health: 40,
        attack: 4,
        movement: 2,
        defense: 5,
        range: 1,
        veteran: false,
        skills: [SkillType.Escape, SkillType.Float]
    },   
    [UnitType.Amphibian]: {
        tribeType: TribeType.Aquarion,
        techType: TechnologyType.Amphibian,
        cost: 3,
        health: 10,
        attack: 2,
        movement: 2,
        defense: 1,
        range: 1,
        skills: [SkillType.Float, SkillType.Dash, SkillType.Escape, SkillType.Fortify]
    },
    [UnitType.Tridention]: {
        tribeType: TribeType.Aquarion,
        techType: TechnologyType.Spearing,
        cost: 8,
        health: 10,
        attack: 2.5,
        movement: 2,
        defense: 1,
        range: 2,
        skills: [SkillType.Float, SkillType.Dash, SkillType.Escape, SkillType.Fortify]
    },

    // Polaris
    [UnitType.Gaami]: {
        tribeType: TribeType.Polaris,
        techType: TechnologyType.Unbuildable,
        cost: 10,
        health: 30,
        attack: 4,
        movement: 1,
        defense: 3,
        range: 1,
        veteran: false,
        skills: [SkillType.AutoFreeze, SkillType.FreezeArea]
    },
    [UnitType.Mooni]: {
        tribeType: TribeType.Polaris,
        techType: TechnologyType.Frostwork,
        cost: 5,
        health: 10,
        attack: 0,
        movement: 1,
        defense: 1,
        range: 1, 
        veteran: false,
        skills: [SkillType.AutoFreeze, SkillType.Skate]
    },
    [UnitType.BattleSled]: {
        tribeType: TribeType.Polaris,
        techType: TechnologyType.Sledding,
        cost: 5,
        health: 15,
        attack: 3,
        movement: 2,
        defense: 2,
        range: 1, 
        skills: [SkillType.Dash, SkillType.Escape, SkillType.Skate]
    },
    [UnitType.IceArcher]: {
        tribeType: TribeType.Polaris,
        techType: TechnologyType.Archery,
        cost: 3,
        health: 10,
        attack: 0.1,
        movement: 1,
        defense: 1,
        range: 2,
        skills: [SkillType.Dash, SkillType.Freeze, SkillType.Fortify]
    },
    [UnitType.IceFortress]: {
        tribeType: TribeType.Polaris,
        techType: TechnologyType.PolarWarfare,
        health: 20,
        cost: 15,
        attack: 4,
        movement: 1,
        defense: 3,
        range: 2,
        veteran: false,
        skills: [SkillType.Skate, SkillType.Scout]
    },

    // Cymanti
    [UnitType.Centipede]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Unbuildable,
        cost: 10,
        health: 20,
        attack: 4,
        movement: 2,
        defense: 3,
        range: 1,
        veteran: false,
        skills: [SkillType.Dash, SkillType.Eat, SkillType.Creep]
    },
    [UnitType.Segment]:     {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Unbuildable,
        cost: -1,
        health: 10,
        attack: 2,
        movement: 1,
        defense: 2,
        range: 1,
        veteran: false,
        explodeOnly: true,
        skills: [SkillType.Independent, SkillType.Creep, SkillType.Explode]
    },
    [UnitType.Doomux]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.ShockTactics,
        cost: 10,
        health: 20,
        attack: 4,
        movement: 3,
        defense: 2,
        range: 1,
        veteran: true,
        skills: [SkillType.Dash, SkillType.Creep, SkillType.Explode]
    },
    [UnitType.Shaman]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Philosophy,
        cost: 10,
        health: 10,
        attack: 1,
        movement: 1,
        defense: 1,
        range: 1,
        veteran: false,
        skills: [SkillType.Convert, SkillType.Boost]
    },
    [UnitType.Kiton]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Strategy,
        cost: 3,
        health: 15,
        attack: 1,
        movement: 1,
        defense: 3,
        range: 1,
        skills: [SkillType.Poison]
    },
    [UnitType.Hexapod]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Riding,
        cost: 3,
        health: 5,
        attack: 3,
        movement: 2,
        defense: 1,
        range: 1,
        skills: [SkillType.Dash, SkillType.Escape, SkillType.Creep]
    },
    [UnitType.Raychi]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Pascetism,
        cost: 8,
        health: 15,
        attack: 3,
        movement: 3,
        defense: 2,
        range: 1,
        skills: [SkillType.Dash, SkillType.Float, SkillType.Creep, SkillType.Navigate, SkillType.Explode]
    },
    [UnitType.Phychi]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Archery,
        cost: 3,
        health: 5,
        attack: 1,
        movement: 2,
        defense: 2,
        range: 2,
        veteran: true,
        skills: [SkillType.Dash, SkillType.Fly, SkillType.Poison, SkillType.Surprise]
    },
    [UnitType.Exida]: {
        tribeType: TribeType.Cymanti,
        techType: TechnologyType.Mathematics,
        cost: 8,
        health: 10,
        attack: 3,
        movement: 1,
        defense: 1,
        range: 3,
        veteran: true,
        skills: [SkillType.Poison, SkillType.Splash]
    },
    
    // Elyron
    [UnitType.DragonEgg]:   {
        tribeType: TribeType.Elyrion,
        techType: TechnologyType.Unbuildable,
        health: 10,
        cost: 10,
        attack: 0,
        movement: 1,
        defense: 2,
        range: 1,
        veteran: false,
        becomes: UnitType.BabyDragon,
        skills: [SkillType.Grow, SkillType.Fortify]
    },
    [UnitType.BabyDragon]:  {
        tribeType: TribeType.Elyrion,
        techType: TechnologyType.Unbuildable,
        health: 15,
        cost: 10,
        attack: 3,
        movement: 2,
        defense: 3,
        range: 1,
        veteran: false,
        becomes: UnitType.FireDragon,
        skills: [SkillType.Grow, SkillType.Dash, SkillType.Fly, SkillType.Escape, SkillType.Scout]
    },
    [UnitType.FireDragon]:  {
        tribeType: TribeType.Elyrion,
        techType: TechnologyType.Unbuildable,
        health: 20,
        cost: 10,
        attack: 4,
        movement: 3,
        defense: 3,
        range: 2,
        veteran: false,
        skills: [SkillType.Dash, SkillType.Fly, SkillType.Splash, SkillType.Scout]
    },
    [UnitType.Polytaur]: {
        tribeType: TribeType.Elyrion,
        techType: TechnologyType.ForestMagic,
        cost: 2,
        health: 15,
        attack: 3,
        movement: 1,
        defense: 1,
        range: 2,
        veteran: false,
        skills: [SkillType.Dash, SkillType.Fortify, SkillType.Independent]
    },
};
