import { getDefenseBonus, getRealUnitSettings, getRealUnitType } from "../core/functions";
import Move from "../core/move";
import { MoveType } from "../core/types";
import { GameState, StructureState, UnitState } from "../core/states";
import { UnitType, TechnologyType, CaptureType } from "../core/types";

export type StagedValue = number[];

/**
 * Minimum amount of moves the AI is forced to play before they can end the turn.
 * NOT PROPERLY IMPLEMENTED
 */
export const MIN_PLAYED_MOVES: StagedValue = [3, 6, 6, 8];

/**
 * Uses slerp to interpolate between the start game and the end game
 * @param state 
 * @param value 
 */
export function lerpViaGameStage(state: GameState, value: StagedValue): number {
    const stage = state.settings._turn / state.settings.maxTurns;
    const t = stage * (value.length - 1);
    const index1 = Math.floor(t);
    const index2 = Math.ceil(t);
    const s = t - index1;
    return value[index1] + s * (value[index2] - value[index1]);
}

/**
 * Returns a calculated score using the unit's class attributes.
 * Naval units yield base class type
 * @param unit 
 */
export function SCORE_UNIT_POWER(state: GameState, unit: UnitState) {
    const settings = getRealUnitSettings(unit);
    
    const bonus = getDefenseBonus(state, unit);
    const { 
        attack, defense, cost, movement, range, 
        skills, becomes, health, 
        veteran, upgradeFrom 
    } = settings;
    const hp = unit._health;
    const isNaval = unit._passenger;

    // effects like poisond, frozen and boosted
    const effect = unit._effects;
}

export const SCORE_UNIT_STRENGTH: Record<UnitType, number> = {
    [UnitType.None]:        0.0,
    // S tier
    [UnitType.Shaman]:      2.5,
    [UnitType.Rider]:       2.0,
    [UnitType.Hexapod]:     1.9,
    [UnitType.BattleSled]:  2.4,
    [UnitType.Amphibian]:   2.2,
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
    [UnitType.Warrior]:     1.0,
    [UnitType.Defender]:    1.5,
    [UnitType.Catapult]:    1.4,
    [UnitType.Kiton]:       1.3,
    [UnitType.Tridention]:  1.4,
    [UnitType.Mooni]:       1.6,
    [UnitType.Raychi]:      1.5,
    [UnitType.Giant]:       10.0,
    [UnitType.Gaami]:       12.0,
    [UnitType.Phychi]:      1.5,
    [UnitType.Exida]:       1.5,
    [UnitType.Doomux]:      14.0,
    // C tier
    [UnitType.Swordsman]:   1.3,
    [UnitType.IceArcher]:   1.4,
    [UnitType.Bomber]:      2.1,
    [UnitType.Juggernaut]:  8.0,
    // F tier
    [UnitType.MindBender]:  0.5,
    [UnitType.IceFortress]: 0.6,
    [UnitType.Raft]:        0.9,
};

/**
 * Score based on how good it is in the game.
 */
export const SCORE_UNIT_TIER: Record<UnitType, number> = {
    [UnitType.None]:        0,
    // S tier
    [UnitType.Shaman]:      3,
    [UnitType.Rider]:       3,
    [UnitType.Hexapod]:     3,
    [UnitType.BattleSled]:  3,
    [UnitType.Amphibian]:   3,
    // A tier
    [UnitType.Archer]:      2,
    [UnitType.Knight]:      2,
    [UnitType.Dinghy]:      2,
    [UnitType.Cloak]:       2,
    [UnitType.Dagger]:      2,
    [UnitType.Pirate]:      2,
    [UnitType.Polytaur]:    2,
    [UnitType.Rammer]:      2,
    [UnitType.Scout]:       2,
    [UnitType.Centipede]:   2,
    [UnitType.Segment]:     2,
    [UnitType.FireDragon]:  2,
    [UnitType.BabyDragon]:  2,
    [UnitType.DragonEgg]:   2,
    [UnitType.Crab]:        2,
    // B tier
    [UnitType.Warrior]:     1,
    [UnitType.Defender]:    1,
    [UnitType.Catapult]:    1,
    [UnitType.Kiton]:       1,
    [UnitType.Tridention]:  1,
    [UnitType.Mooni]:       1,
    [UnitType.Raychi]:      1,
    [UnitType.Giant]:       1,
    [UnitType.Gaami]:       1,
    [UnitType.Phychi]:      1,
    [UnitType.Exida]:       1,
    [UnitType.Doomux]:      1,
    // C tier
    [UnitType.Swordsman]:   0,
    [UnitType.IceArcher]:   0,
    [UnitType.Bomber]:      0,
    [UnitType.Juggernaut]:  0,
    // F tier
    [UnitType.MindBender]:  -1,
    [UnitType.IceFortress]: -1,
    [UnitType.Raft]:        -1,
};

/**
 * Rewarded more on the type of unit and how close it is to enemy city borders
 */
export const CAPTURE_POTENTIAL: Record<UnitType, number> = {
    // S
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
    [UnitType.Catapult]:    0.1,
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

// TODO
/**
 * Simple heuristic to weigh a move. EndTurn is always mid priority.
 * @param move 
 * @returns 
 */
export function SCORE_MOVE_PRIORITY(move: Move): number {
    switch (move.moveType) {
        case MoveType.Capture:
            const captureType = move.getType<CaptureType>();
            if(captureType == CaptureType.City) {
                return 20;
            }
            else if(captureType == CaptureType.Village) {
                return 10;
            }
            else if(captureType == CaptureType.Ruins || captureType == CaptureType.Starfish) {
                return 10;
            }
            else {
                return 9;
            }
        case MoveType.Ability:
            return 9;
        case MoveType.Step:
            return 7;
        case MoveType.Attack:
            return 6;
        case MoveType.Summon:
            return 5;
        case MoveType.Harvest:
            return 7;
        case MoveType.Build:
            return 7;
        case MoveType.Research:
            return 15;
    }
    return 1;
}

Object.freeze(SCORE_UNIT_STRENGTH);
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