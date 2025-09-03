import { getDefenseBonus, getPovTerritorry, getPovTribe, getRealUnitSettings, getRealUnitType, getTribeSPT } from "../core/functions";
import Move from "../core/move";
import { MoveType } from "../core/types";
import { GameState, StructureState, UnitState } from "../core/states";
import { UnitType, TechnologyType, CaptureType } from "../core/types";
import { GMath } from "./gmath";

export type StageValue = number[];

/**
 * Minimum amount of moves the AI is forced to play before they can end the turn.
 * NOT PROPERLY IMPLEMENTED
 */
export const MIN_PLAYED_MOVES: StageValue = [3, 6, 6, 8];

/**
 * The weight of the economy and army score depending on the current stage of the game.
 * EarlyGame, MidGame, EndGame
 */
export const STAGE_SCORE: [StageValue, StageValue, StageValue] = [
    [1.0, 0.8],
    [0.9, 1.0],
    [0.5, 1.0],
];

/**
 * The thresholds that determine the stages of the game from the start to end.
 * Adds up to ~1.
 */
export const STAGE_THRESH: [number, number, number] = [
    0.333,
    0.333,
    0.333,
];

/**
 * TODO NOT IMPLEMENTED WITH STAGE_THRESH
 * Uses slerp to interpolate between the start game and the end game
 * @param state 
 * @param value 
 */
export function lerpViaGameStage(state: GameState, value: StageValue): number {
    const stage = state.settings._turn / state.settings.maxTurns;
    const t = stage * (value.length - 1);
    const index1 = Math.floor(t);
    const index2 = Math.ceil(t);
    const s = t - index1;
    return value[index1] + s * (value[index2] - value[index1]);
}

/**
 * Returns a score based on the unit's current on-field attributes.
 * Naval units inherit original their base class type.
 * @param unit 
 */
export function SCORE_UNIT_POWER(state: GameState, unit: UnitState) {
    let score = 0;
    const settings = getRealUnitSettings(unit);
    const unitType = getRealUnitType(unit);
    const isNaval = Boolean(unit._passenger);

    // 1.0x (default), 1.5x (some units like defender), or 4.0x (unit in city walls)
    const defBonus = getDefenseBonus(state, unit);

    const { 
        attack, defense, cost, movement, range, 
        skills: skillTypes, health: maxHealth, 
        veteran: isVeteran
    } = settings;

    const hp = unit._health;

    // EffectType, poisoned, frozen and boosted
    // poisoned reduces def to 1.0 regardless defBonus
    // boosted gives +0.5 atk
    // frozen units cant move or do anything until and a turn is wasted
    const effect = unit._effects;

    return score;
}

/**
 * Multiplier on generally how worth it or good the unit is.
 */
export const SCORE_UNIT_STRENGTH: Record<UnitType, number> = {
    [UnitType.None]:        0.0,
    // Super Units
    // S tier
    [UnitType.Shaman]:      15.0,
    // A tier
    [UnitType.FireDragon]:  10.0,
    [UnitType.BabyDragon]:  8.0,
    [UnitType.DragonEgg]:   7.0,
    [UnitType.Centipede]:   9.0,
    [UnitType.Segment]:     5.0,
    [UnitType.Crab]:        9.0,
    // B tier
    [UnitType.Giant]:       7.0,
    [UnitType.Gaami]:       7.0,
    // C tier
    [UnitType.Juggernaut]:  6.0,

    // Spawnable Units
    // S tier
    [UnitType.Rider]:       3.2,
    [UnitType.Hexapod]:     3.1,
    [UnitType.BattleSled]:  3.0,
    [UnitType.Amphibian]:   3.0,
    // A tier
    [UnitType.Archer]:      2.7,
    [UnitType.Knight]:      2.6,
    [UnitType.Cloak]:       2.5,
    [UnitType.Dinghy]:      2.3,
    [UnitType.Dagger]:      2.3,
    [UnitType.Pirate]:      2.3,
    [UnitType.Polytaur]:    2.2,
    [UnitType.Rammer]:      2.1,
    [UnitType.Scout]:       2.0,
    // B tier
    [UnitType.Warrior]:     1.9,
    [UnitType.Defender]:    1.8,
    [UnitType.Catapult]:    1.7,
    [UnitType.Kiton]:       1.6,
    [UnitType.Tridention]:  1.6,
    [UnitType.Mooni]:       1.6,
    [UnitType.Raychi]:      1.5,
    [UnitType.Phychi]:      1.5,
    [UnitType.Exida]:       1.5,
    [UnitType.Doomux]:      1.4,
    // C tier
    [UnitType.Swordsman]:   0.9,
    [UnitType.IceArcher]:   0.9,
    [UnitType.Bomber]:      0.7,
    // F tier
    [UnitType.MindBender]:  0.5,
    [UnitType.IceFortress]: 0.6,
    [UnitType.Raft]:        0.6,
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
    [UnitType.Swordsman]:   0.5,
    [UnitType.IceArcher]:   0.5,
    [UnitType.Bomber]:      0.5,
    [UnitType.Juggernaut]:  0.5,
    // F tier
    [UnitType.MindBender]:  0,
    [UnitType.IceFortress]: 0,
    [UnitType.Raft]:        0,
};

/**
 * Multiplier on how effective it is closer to enemy city borders
 */
export const CAPTURE_POTENTIAL: Record<UnitType, number> = {
    [UnitType.None]:        0.0,
    [UnitType.Crab]:        5.5,
    [UnitType.Juggernaut]:  5.4,
    [UnitType.Giant]:       5.0,
    [UnitType.Gaami]:       4.0,
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
    [UnitType.DragonEgg]:   1.0,
    [UnitType.BabyDragon]:  1.0,
    [UnitType.FireDragon]:  1.0,
    [UnitType.Doomux]:      1.0,
    [UnitType.Phychi]:      1.0,
    [UnitType.Exida]:       1.0,
    [UnitType.Centipede]:   1.0,
    [UnitType.Segment]:     1.0,
};

/**
 * TODO NOT PEOPERLY IMPLEMENTED (REVISE)
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
            return 5;
    }
    return 1;
}

/**
 * Evaluates the established economy for the POV tribe and returns a score.
 * @param state 
 */
export function evaluateEconomy(state: GameState): number {    
    let score = 0;

    let spt = getTribeSPT(state);
    // Cap the SPT with a maximum ideal
    spt = GMath.clamp(spt, 60) / 60;

    score += spt;
    
    return score;
}

/**
 * Evaluates the military strength for the POV tribe and returns a score.
 * @param state 
 */
export function evaluateArmy(state: GameState) {
    let score = 0;
    const tribe = getPovTribe(state);
    
    // Military strength:
    // amount of units
    // type of units
    // map control (defense & offense)

    const army = tribe._units;

    let rawUnitStrength = 0;

    for (let u = 0; u < army.length; u++) {
        const unit = army[u];

        rawUnitStrength += SCORE_UNIT_TIER[unit._unitType];
        rawUnitStrength += SCORE_UNIT_STRENGTH[unit._unitType];

        rawUnitStrength += SCORE_UNIT_POWER(state, unit);
        
    }

    return score;
}

/**
 * Evaluates the entire state of the game for the POV tribe and returns a score.
 * @param state 
 */
export function evaluateState(state: GameState): [number, number, number] {
    let eco = evaluateEconomy(state);
    let army = evaluateArmy(state);
    let score = 0;

    return [eco, army, score];
}

Object.freeze(SCORE_UNIT_STRENGTH);
Object.freeze(STAGE_SCORE);
Object.freeze(STAGE_THRESH);
Object.freeze(SCORE_UNIT_TIER);
Object.freeze(CAPTURE_POTENTIAL);