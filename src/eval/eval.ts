import { GameState, StructureState, UnitState } from "../core/states";
import { UnitType, TechnologyType, CaptureType } from "../core/types";


const UNIT_SCORES: Record<UnitType, number> = {
    [UnitType.None]:        0.0,
    // S tier
    [UnitType.Shaman]: 2.5,
    [UnitType.Rider]: 2.0,
    [UnitType.Hexapod]: 1.9,
    // TODO Jelly (aquarion)
    [UnitType.BattleSled]: 2.3,
    // A tier
    [UnitType.Archer]:      1.4,
    [UnitType.Knight]:      1.7,
    [UnitType.Dinghy]:      1.9,
    [UnitType.Cloak]:       1.8,
    [UnitType.Dagger]:      1.1,
    [UnitType.Pirate]:      1.0,
    [UnitType.Polytaur]:    1.75,
    [UnitType.Rammer]:      1.6,
    [UnitType.Scout]:       1.6,
    [UnitType.Centipede]:   15.0,
    [UnitType.Segment]:     2.0,
    [UnitType.FireDragon]:  18.0,
    [UnitType.BabyDragon]:  6.0,
    [UnitType.DragonEgg]:   5.0,
    [UnitType.Crab]:        16.0,
    // B tier
    [UnitType.Warrior]: 1.2,
    [UnitType.Defender]: 1.5,
    [UnitType.Catapult]: 1.4,
    [UnitType.Kiton]: 1.3,
    [UnitType.Tridention]: 1.4,
    [UnitType.Mooni]: 1.6,
    [UnitType.Raychi]: 1.5,
    [UnitType.Giant]: 10.0,
    [UnitType.Gaami]: 12.0,
    [UnitType.Phychi]: 1.5,
    [UnitType.Exida]: 1.5,
    [UnitType.Doomux]: 14.0,
    // C tier
    [UnitType.Swordsman]: 1.3,
    [UnitType.IceArcher]: 1.4,
    [UnitType.Bomber]: 2.1,
    [UnitType.Juggernaut]: 8.0,
    // TODO Shark
    // F tier
    [UnitType.MindBender]: 0.5,
    [UnitType.IceFortress]: 0.6,
    [UnitType.Raft]: 0.9,
    // TODO Pufferfish
    // Unclasified
    [UnitType.Amphibian]: 1.6,
};

const CAPTURE_POTENTIAL: Record<UnitType, number> = {
    [UnitType.None]:        0.0,
    [UnitType.Gaami]:       6.0,
    [UnitType.Crab]:        5.5,
    [UnitType.Juggernaut]:  5.4,
    [UnitType.Giant]:       5.0,
    [UnitType.Shaman]:      4.0,
    [UnitType.Dinghy]:      2.6,
    [UnitType.Cloak]:       2.5,
    [UnitType.Dagger]:      1.7,
    [UnitType.Pirate]:      1.4,
    [UnitType.Hexapod]:     2.5,
    [UnitType.Amphibian]:   2.2,
    [UnitType.Rider]:       2.0,
    [UnitType.Tridention]:  2.0,
    [UnitType.Swordsman]:   1.7,
    [UnitType.BattleSled]:  1.6,
    [UnitType.Knight]:      1.4,
    [UnitType.Polytaur]:    1.3,
    [UnitType.Warrior]:     1.2,
    [UnitType.Defender]:    1.1,
    [UnitType.Rammer]:      1.1,
    [UnitType.Scout]:       1.1,
    [UnitType.Archer]:      0.7,
    [UnitType.Bomber]:      0.6,
    [UnitType.Catapult]:    0.3,
    [UnitType.MindBender]:  0.5,
    [UnitType.Mooni]:       0.6,
    [UnitType.IceFortress]: 0.9,
    [UnitType.IceArcher]:   0.6,
    [UnitType.Kiton]:       1.1,
    [UnitType.Raychi]:      1.5,
    [UnitType.Raft]:        0.6,
    [UnitType.DragonEgg]:   0,
    [UnitType.BabyDragon]:  0,
    [UnitType.FireDragon]:  0,
    [UnitType.Doomux]:      0,
    [UnitType.Phychi]:      0,
    [UnitType.Exida]:       0,
    [UnitType.Centipede]:   0,
    [UnitType.Segment]:     0,
};

Object.freeze(UNIT_SCORES);
Object.freeze(CAPTURE_POTENTIAL);

function calculateDistance(tileIndex1: number, tileIndex2: number, size: number) {
    return 0;
}

export function rewardUnitMove(state: GameState, unit: UnitState, ipX: number, ipY: number): number {    
    return 0;
}

export function rewardUnitAttack(state: GameState, unit: UnitState, enemy: UnitState): number {    
    return 0;
}

export function rewardCapture(state: GameState, unit: UnitState, captureType: CaptureType): number {    
    return 0;
}

export function rewardTech(state: GameState, techType: TechnologyType): number {    
    return 0;
}

export function rewardStructure(state: GameState, struct: StructureState): number {    
    return 0;
}

/**
 * Evaluate the state of the economy.
 * @param state 
 * @param depth 
 * @returns 
 */
export function evaluateEconomy(state: GameState) {    
    return 0;
}

/**
 * Evaluate how good spawning a unit is.
 * @param state 
 * @param tileIndex 
 * @param unitType 
 * @returns 
 */
export function evaluateUnitSpawn(state: GameState, tileIndex: number, unitType: UnitType): number {    
    return 0;
}

/**
 * Evaluate the state of the army.
 * @param state 
 * @returns 
 */
export function evaluateArmy(state: GameState) {    
    return 0;
}

/**
 * Reward and penalize by enemy units in visible region.
 * @param state
 * @returns number
 */
function scoreVisibleEnemyUnits(state: GameState): number {    
    return 0;
}