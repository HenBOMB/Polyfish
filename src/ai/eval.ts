import { getClosestEnemyCity, getDefenseBonus, getPovTerritorry, getPovTribe, getRealUnitSettings, getRealUnitType, getTribeSPT, hasEffect, isAdjacentToEnemy, getAdjacentTiles, getAdjacentIndexes, calculateTribeScore } from "../core/functions";
import Move from "../core/move";
import { EffectType, ModeType, MoveType } from "../core/types";
import { GameState, UnitState } from "../core/states";
import { CaptureType } from "../core/types";
import { GMath } from "./gmath";
import Game from "../game";

// export type StageValue = [number, number];


/**
 * The maximum ideal score an AI can get in the 30 turns perfection gamemode.
 */
const MAX_IDEAL_SCORE = 100_000;

/**
 * Minimum amount of moves the AI is forced to play before they can end the turn.
 * NOT PROPERLY IMPLEMENTED
 */
export const MIN_PLAYED_MOVES = [3, 6, 6, 8];

/**
 * The weight of the economy and army score depending on the current stage of the game.
 * [EarlyGame, MidGame, EndGame]
 * [[Economy, Army], ...]
 * Values always adds up to 1.
 */
const STAGE_SCORE_PERFECTION: [[number, number], [number, number], [number, number]] = [
    GMath.probabilities(GMath.zeros(2), 0, 0.90),
    GMath.probabilities(GMath.zeros(2), 0, 0.85),
    GMath.probabilities(GMath.zeros(2), 0, 0.70),
];
// ! [[0.9, ??], [0.85, ??], [0.70, ??]]
// ? focus more on the economy rather than the military
STAGE_SCORE_PERFECTION.forEach(arr => Object.freeze(arr));
Object.freeze(STAGE_SCORE_PERFECTION);
// TODO also for Domination

/**
 * The thresholds that determine the stages of the game from the start to end.
 * Always adds up to 1.
 */
const STAGE_THRESH: [number, number, number] = [
    0.2, 
    1 - (0.2 + 0.2), 
    0.2
];
// ? the early game lasts 20% of the first turns, and the endgame 20% of the last turns
Object.freeze(STAGE_THRESH);

/**
 * Calculates the current economy and army multiplies depensing on the game stage
 * @param state 
 * @returns [EconomyMult, ArmyMult]
 */
export function calculateStageValue(state: GameState): [number, number] {
    const p = state.settings._turn / state.settings.maxTurns;

    if(p < STAGE_THRESH[0]) {
        return STAGE_SCORE_PERFECTION[0];
    }
    else if(p < STAGE_THRESH[1]) {
        return STAGE_SCORE_PERFECTION[1];
    }
    else {
        return STAGE_SCORE_PERFECTION[2];
    }
}

/**
 * @Obsolete
 * TODO NOT IMPLEMENTED WITH STAGE_THRESH
 * Uses slerp to interpolate between the start game and the end game
 * @param state 
 * @param value 
 */
export function lerpViaGameStage(state: GameState, value: number[]): number {
    const stage = state.settings._turn / state.settings.maxTurns;
    const t = stage * (value.length - 1);
    const index1 = Math.floor(t);
    const index2 = Math.ceil(t);
    const s = t - index1;
    return value[index1] + s * (value[index2] - value[index1]);
}

/**
 * Returns a sort-of normalized score based on the unit's current on-field attributes.
 * This acts as a multiplier on the unit's base strength.
 * @param state 
 * @param unit 
 */
export function scoreUnitPower(game: Game, unit: UnitState): number {
    const state = game.state;
    const settings = getRealUnitSettings(unit);
    
    const n_values = 2;

    // A unit's value is directly proportional to its health, a 1hp unit is far less valuable
    // Even its atk is proportional
    const healthMultiplier = unit._health / settings.health!;

    // A unit in a walled city is much stronger
    // 1.0 (normal), 1.5 (forest, water, etc), 4.0 (city wall)
    // A unit's maximum defense is 5.0 and bonus is 4.0
    const defenseValue = (getDefenseBonus(state, unit) * settings.defense) / (4.0 * 5.0); 

    // Status effects
    let effectMultiplier = 1.0;
    if (hasEffect(unit, EffectType.Poison)) {
        // Poison is bad, reduces survivability
        effectMultiplier *= 0.7; 
    }
    if (hasEffect(unit, EffectType.Boost)) {
        // Boost is a significant advantage cause it boosts attack and defense
        effectMultiplier *= 1.2; 
    }
    // Frozen is complete trash, renders the unit useless for an entire turn
    if (hasEffect(unit, EffectType.Frozen)) {
        effectMultiplier *= 0.1;
    }
    // Having invisibility is pointless if we are not using it (cloaks)
    if (hasEffect(unit, EffectType.Invisible)) {
        // TODO if the unit has an adjacent enemy unit, then its bad because they can see partially see them!
        // and potentially get revealed and insta-killed

        // But dont include invisible units because we cant actually see them and there is no indicator they are there or can see us.
        // (eye icon on the enemy unit reveals this so its not cheating)
        if (isAdjacentToEnemy(state, state.tiles[unit._tileIndex], undefined, false)) {
            effectMultiplier *= 0.5; 
        }
        else {
            effectMultiplier *= 1.3; 
        }
    }

    // Veteran units are more resilient and are worth keeping alive
    const veteranBonus = 1;//unit._veteran ? 1.15 : 1.0;
    // Disabled because it only boosts the health and restores it completely, which is handled by healthMultiplier

    // Combine everything
    let finalScore = healthMultiplier * defenseValue * effectMultiplier * veteranBonus;

    finalScore *= game.values.unitStrength.get(unit._unitType);

    return finalScore / n_values;
}

/**
 * TODO NOT PEOPERLY IMPLEMENTED (REVISE)
 * Simple heuristic to weigh a move. EndTurn is always mid priority.
 * @param move 
 * @returns 
 */
export function scoreMovePriority(move: Move): number {
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
 * @returns A number between 0 and 1
 */
export function evaluateEconomy(state: GameState): number {    
    let score = 0;

    let spt = getTribeSPT(state);

    // Cap the SPT with a maximum ideal
    spt = GMath.clamp(spt, 60) / 60;

    score += spt;

    // Add the tribe's score with some maximum if we're playing `Perfection` mode
    if (state.settings.mode === ModeType.Perfection) {

        score += GMath.clamp(getPovTribe(state)._score, MAX_IDEAL_SCORE) / MAX_IDEAL_SCORE;

        // 2 is to normalize the value to 0-1, since we've added the spt (1) and tribe score (2)
        return score / 2;
    }

    return score;
}

/**
 * Evaluates the military strength for the POV tribe and returns a score.
 * @param state 
 * @returns A number between 0 and 1
 */
export function evaluateArmy(game: Game): number {
    const state = game.state;
    const povTribe = getPovTribe(state);
    // const povTerritory = getPovTerritorry(state);

    let armyScore = 0;
    let captureScore = 0;
    let tilesControlled = new Set();

    // If we dont add this then at lower unit amounts there will be biased higher scores
    const maxUnitCount = 20;
    // It may skip important units if the tribe has more than this amount, if so, just increase this value

    // Calculate the total army score
    for (let i = 0; i < GMath.clamp(povTribe._units.length, maxUnitCount); i++) {
        const unit = povTribe._units[i];

        // Army Strength //

        // Calculate raw power on the current battlefield
        armyScore += scoreUnitPower(game, unit);

        // Calculate map control and positioning
        const control = getAdjacentIndexes(state, unit._tileIndex);

        for (let i = 0; i < control.length; i++) {
            if (!tilesControlled.has(state.tiles[control[i]].tileIndex)) {
                tilesControlled.add(state.tiles[control[i]].tileIndex);
            }
        }

        // Capture Potential //

        // Search in a maximum of 6 tiles
        const closestData = getClosestEnemyCity(state, unit._tileIndex, 6);
        
        // If there are no cities, then there will never be one for any of our units
        if (!closestData) {
            break;
        }

        const tile = state.tiles[closestData[0].tileIndex];
        // Add a score bonus for being close, the bonus diminishes with distance
        // Add a small bonus if its a capital
        captureScore += GMath.clamp(
            game.values.capturePotential.get(getRealUnitType(unit)) 
                + (tile.capitalOf > 0? 0.2 : 0) ,
            1
        ) / (closestData[1] + 1);
    }

    // Control maximally 80% of the map
    const maxControl = state.settings.size * 0.8;
    
    let controlScore = GMath.clamp(tilesControlled.size, maxControl);

    // Normalize all scores
    armyScore /= maxUnitCount;
    captureScore /= maxUnitCount;
    controlScore /= maxControl;

    // The final army score is a combination of the raw power of our units and their strategic positioning
    const finalScore = 
        0.6 * armyScore + 
        0.15 * captureScore + 
        0.25 * controlScore;

    return finalScore;
}

/**
 * Evaluates the entire state of the game for the POV tribe and returns a score.
 * @param state 
 */
export function evaluateState(game: Game): [number, number, number] {
    const state = game.state;
    
    const pov = state.settings._pov;
    const [ecoMult, armyMult] = calculateStageValue(state);

    const myEcoScore = evaluateEconomy(state);
    const myArmyScore = evaluateArmy(game);
    const myScore = 
        ecoMult * myEcoScore + 
        armyMult * myArmyScore;

    // Going with perfection gamemode for now, simpler but essential for the future Domination gamemode
    // We should only get the opponent's top score and use that as reference

    let theirScore = 0;
    let theirEcoScore = 0;
    let theirArmyScore = 0;
    let theirTribeScore = 0;

    for (const owner in state.tribes) {
        if (state.settings._pov === Number(owner)) {
            continue;
        }
        
        state.settings._pov = Number(owner);
        const _theirEcoScore = evaluateEconomy(state);
        const _theirArmyScore = evaluateArmy(game);
        const curScore = 
            ecoMult * _theirEcoScore + 
            armyMult * _theirArmyScore;

        if (curScore > theirScore) {
            theirScore = curScore;
            theirTribeScore = getPovTribe(state)._score;
            theirEcoScore = _theirEcoScore;
            theirArmyScore = _theirArmyScore;
        }
    }

    state.settings._pov = pov;

    // A 0 value means there are no opponents left
    // if (theirScore !== 0) {
    //     theirScore = GMath.clamp(theirScore, MAX_IDEAL_SCORE) / MAX_IDEAL_SCORE;
    // }
    
    // TODO use stage values?

    // console.log('\n--Evaluation--')
    // console.log(`my tribe score: ${getPovTribe(state)._score}`);
    // // console.log(`my eco score: ${myEcoScore.toFixed(4)}`);
    // // console.log(`my army score: ${myArmyScore.toFixed(4)}`);
    // console.log(`their tribe score: ${theirTribeScore}`);
    // // console.log(`their eco score: ${theirEcoScore.toFixed(4)}`);
    // // console.log(`their army score: ${theirArmyScore.toFixed(4)}`);
    // console.log(`my score: ${myScore.toFixed(4)}`);
    // console.log(`their score: ${theirScore.toFixed(4)}`);

    // 0.1 is a small boost to our own score
    const finalScore = (myScore - theirScore);// + (myScore * 0.01)

    // console.log(`net score: ${finalScore}`);

    return [
        myEcoScore, 
        myArmyScore, 
        finalScore
    ];
}
