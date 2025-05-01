import { calculateDistance, getAlliesNearTile, getClosestOwnedStructureTile, getPovTribe, getTechCost, isAquaticOrCanFly } from "../core/functions";
import Move, { MoveType } from "../core/move";
import UnitMoveGenerator, { generateEcoMoves, generateEndTurnMove, generateResourceMoves, generateStructureMoves, generateTechMoves, UndoCallback } from "../core/moves";
import { ResourceSettings } from "../core/settings/ResourceSettings";
import { StructureSettings } from "../core/settings/StructureSettings";
import { TechnologySettings } from "../core/settings/TechnologySettings";
import { UnitSettings } from "../core/settings/UnitSettings";
import { CityState, GameState, TribeState, UnitState } from "../core/states";
import { ResourceType, RewardType, StructureType, TechnologyType, UnitType } from "../core/types";
import { evaluateEconomy, evaluateArmy, evaluateUnitSpawn } from "./eval";
import { findBestMoves } from "./mctsgpt";

export interface BestMoves {
    moves: Move[];
    score: number;
}

export function logAndUndoMoves(moves: Move[], state: GameState, verbose = true, undoChain?: UndoCallback[]): {
    starsLeft: number;
    scoreEconomy: number;
    scoreArmy: number;
} {
    const initialEco = evaluateEconomy(state);
    const initialArmy = evaluateArmy(state);
    let prevEco = initialEco;
    let prevArmy = initialArmy;
    const undos: UndoCallback[] = [];
    let starsLeft = getPovTribe(state)._stars;

    const computeDirection = (fromIndex: number, toIndex: number, noidea?: boolean) => {
        const fromX = fromIndex % state.settings.size;
        const fromY = Math.floor(fromIndex / state.settings.size);
        const toX = toIndex % state.settings.size;
        const toY = Math.floor(toIndex / state.settings.size);
        const xDiff = toX - fromX;
        const yDiff = toY - fromY;
        if(((yDiff > 0 && xDiff > 0) || (yDiff < 0 && xDiff < 0)) && xDiff == yDiff) {
            if(yDiff > 0 && xDiff > 0) {
                return noidea? 'Top' : 'Up';
            }
            return noidea? 'Bottom' : 'Down';
        } else if(yDiff > 0 && xDiff < 0) {
            return "Left";
        } else if (xDiff === 0 && yDiff !== 0) {
            if(yDiff > 0) {
                return noidea? 'Top Left' : 'Up Left';
            }
            else {
                return noidea? 'Bottom Right' : 'Down Right';
            }
        } else if (yDiff === 0 && xDiff !== 0) {
            return xDiff < 0 ? noidea? 'Bottom Left' : 'Down Left' : noidea? 'Top Right' : 'Up Right';
        } else if (xDiff !== 0 && yDiff !== 0) {
            const yDir = yDiff < 0 && xDiff < 0 ?
                noidea? 'Bottom' : 'Down' : yDiff > 0? noidea? 'Top' : 'Up' : '';
            const xDir = xDiff < 0 ? ` Left` : yDiff < 0? ' Right' : '';
            return `${yDir}${xDir}`.trim();
        } else {
            return 'Stay';
        }
    }

    for(let i = 0; i < moves.length; i++) {
        const move = moves[i];
        const undo = move.execute(state);
        const ecoScore = evaluateEconomy(state);
        const armyScore = evaluateArmy(state);
        const resultEco = ecoScore - prevEco;
        const resultArmy = armyScore - prevArmy;
        const valueSrc = move.src;
        const valueTarget = move.target;
        const valueType = move.type;

        let log = `${(resultEco === 0 ? '\x1b[90m ' : resultEco >= 0 ? '\x1b[32m+' : '\x1b[31m') + resultEco.toFixed(2)}\x1b[0m,`
              + ` ${(resultArmy === 0 ? '\x1b[90m' : resultArmy >= 0 ? '\x1b[32m+' : '\x1b[31m') + resultArmy.toFixed(2)}\x1b[0m `;

        const moveName = MoveType[move.moveType];

        if(verbose) {
            switch (move.moveType) {
                case MoveType.Step: {
                    const unitType = getPovTribe(state)._units.find(x => x._tileIndex == valueSrc)!._unitType!;
                    const toPort = state.structures[valueTarget]?.id == StructureType.Port && !isAquaticOrCanFly(unitType);
                    log += `${moveName} ${UnitType[unitType]} ${computeDirection(valueSrc, valueTarget)} ${toPort? 'to Port ' : ''}(${valueTarget})`;
                    break;
                }
                case MoveType.Attack: {
				    // id: `_owner, _unitType, _tileIndex, _owner, _unitType, _tileIndex`,
                    const unitType = getPovTribe(state)._units.find(x => x._tileIndex == valueSrc)?._unitType;
                    const enemyType = Object.values(state.tribes).map(x => x.owner != state.settings._pov? x._units : []).flat().find(x => x._tileIndex == valueTarget)?._unitType;
                    log += `Attack ${unitType && UnitType[unitType]} ${`Attack ${enemyType && UnitType[enemyType]}`} ${computeDirection(valueSrc, valueTarget, true)}`;
                    break;
                }
                case MoveType.Summon: {
                    const unitType = UnitType[valueType];
                    const cityName = getPovTribe(state)._cities.find(x => x.tileIndex == valueTarget)!.name;
                    log += `${moveName} ${unitType} on ${cityName}`;
                    break;   
                }
                case MoveType.Research: {
                    log += `${moveName} ${TechnologyType[valueType]}`;
                    break;
                }
                case MoveType.Harvest: {
                    const cityIndex = state.tiles[valueSrc]._rulingCityIndex;
                    const cityName = getPovTribe(state)._cities.find(x => x.tileIndex == cityIndex)!.name;
                    log += `${moveName} ${StructureType[valueType]} ${computeDirection(cityIndex, valueSrc, true)} on ${cityName}`;
                    break;
                }
                case MoveType.Build: {
                    const cityIndex = state.tiles[valueSrc]._rulingCityIndex;
                    const type = StructureType[state.structures[valueSrc]?.id!];
                    const cityName = getPovTribe(state)._cities.find(x => x.tileIndex == cityIndex)!.name;
                    log += `${moveName} ${type} ${computeDirection(cityIndex, valueSrc, true)} on ${cityName}`;
                    break;
                }
                case MoveType.Reward: {
                    log += `${moveName} ${RewardType[valueType]}`;
                    break;
                }
                case MoveType.EndTurn: {
                    log += `End Turn`;
                    break;
                }
                case MoveType.Capture: {
                    const closestVillage = getClosestOwnedStructureTile(state, valueSrc, StructureType.Village)![0];
                    const cityName = getPovTribe(state)._cities.find(x => x.tileIndex == closestVillage)!.name;
                    log += `${moveName} village ${computeDirection(closestVillage, valueSrc, true)} of ${cityName}`;
                    break;
                }
                default:
                    log += `Missing ${move.moveType}`;
                    break;
            }
        }

        if(i == moves.length - 1) {
            starsLeft = getPovTribe(state)._stars;
        }

        verbose && console.log(log);

        prevEco = ecoScore;
        prevArmy = armyScore;
        if(!undoChain) undos.push(undo.undo);
        else undoChain.push(undo.undo);
    }

    if(verbose && moves.length > 1) console.log(`Final score: ${prevEco.toFixed(2)} (${(prevEco - initialEco) >= 0 ? '\x1b[32m+' : '\x1b[31m'}${(prevEco - initialEco).toFixed(2)}\x1b[0m), ${prevArmy.toFixed(2)} (${(prevArmy - initialArmy) >= 0 ? '\x1b[32m+' : '\x1b[31m'}${(prevArmy - initialArmy).toFixed(2)}\x1b[0m)`);
   
    if(!undoChain) undos.reverse().forEach(x => x());

    return {
        starsLeft,
        scoreEconomy: prevEco,
        scoreArmy: prevArmy
    }
}

export function evaluateBestMove(
    state: GameState,
    generator: (state: GameState) => Move[] = generateEcoMoves,
    evaluator: (state: GameState) => number = evaluateEconomy,
    depth = 1
): BestMoves | null {
    const moves = generator(state);
    let bestMove: BestMoves | null = null;

    for (const move of moves) {
        const undoChain = [];
        const br = move.execute(state);
        undoChain.push(br.undo);

        if(br.chainMoves?.length) {
            for (const chainMove of br.chainMoves) {
                const br = chainMove.execute(state);
                undoChain.push(br.undo);
            }
        }

        let score: number;
        let newMoves: Move[];

        if (depth > 0) {
            const subBestMove = evaluateBestMove(state, generator, evaluator, depth - 1);
            if (subBestMove) {
                score = subBestMove.score;
                newMoves = [move, ...(br.chainMoves? br.chainMoves : []), ...subBestMove.moves];
            } else {
                score = evaluator(state);
                newMoves = [move, ...(br.chainMoves? br.chainMoves : [])];
            }
        } else {
            score = evaluator(state);
            newMoves = [move, ...(br.chainMoves? br.chainMoves : [])];
        }

        if (!bestMove || bestMove.score < score) {
            bestMove = { moves: newMoves, score };
        }

        undoChain.reverse().forEach(x => x());
    }

    return bestMove;
}


const generateBestUnitSpawn = (state: GameState, city?: CityState) => {
    const moves = UnitMoveGenerator.spawns(state, city);
    let bestMove = null;
    let bestScore = 0;
    
    for (let i = 0; i < moves.length; i++) {
        const move = moves[i];
        const br = move.execute(state);
        // const endCb = generateEndTurnMove(state).execute(state).undo;

        // const unit = state.tribes[state.settings.pov]._units.find(x => x._tileIndex == tileIndex)!;
        // // const nearbyUnits = getAlliesNearTile(state, unit._tileIndex, 1);

        // biased
        // generate best moves, in a short sequence
        // const bestSubMoves = findBestMoves(state, 10, 4, undefined, (state: GameState) => {
        //     return [
        // //         ...nearbyUnits.map(unit => UnitMoveGenerator.all(state, unit)).flat(),
        //         ...UnitMoveGenerator.all(state, unit),
        //         ...(unit._moved || unit._attacked? [generateEndTurnMove(state)] : []),
        //     ];
        // }, evaluateArmy);

        // spawn-unitType-unitIndex
        const score = evaluateUnitSpawn(state, move.src, move.type);
            // + (bestSubMoves?.score || 0);
            
        if(score > bestScore) {
            bestMove = move
            bestScore = score;
        }
        
        // UnitType[unitType] != 'Warrior' && console.log(UnitType[unitType], Math.round(score * 100) / 100);

        // endCb();
        br.undo();
    }
    return bestMove;
}

export function generateBestTechMove(state: GameState, ...notLike: TechnologyType[]) {
    const tribe = getPovTribe(state);

    // Evaluates if we can also buy whatever this tech unlocks.
    const isTechWorthy = (techType: TechnologyType) => {
        if(notLike.includes(techType)) return false;

        const settings = TechnologySettings[techType];
        const techCost = getTechCost(tribe, techType);

        let consumeCost = 0;
        if(settings.unlocksStructure) {
            consumeCost = StructureSettings[settings.unlocksStructure].cost || 0;
        }
        else if(settings.unlocksResource) {
            consumeCost = ResourceSettings[settings.unlocksResource].cost || 0;
        }
        if(settings.unlocksUnit) {
            consumeCost += UnitSettings[settings.unlocksUnit].cost;
        }

        // We need to consume at least 2
        consumeCost *= 1.6;

        // If feeling risky
        // const spt = tribe._cities.reduce((x, y) => x + y._production, 0);
        // const required = (techCost + consumeCost - tribe.stars) - spt;
        const required = techCost + consumeCost - tribe._stars;

        return required < 1;
    }

    // const bestPossibleMoves: { [id: string]: (string|number)[][] } = {};
    const bestCityMoves: { [cityName: string]: Move[] } = {};

    let bestTechMove: Move | null = null;

    // If we can afford buying this tech and more (3 stars), then we have enough to spend on tech
    const hasEnoughResources = tribe._tech.some(x => getTechCost(tribe, x) <= tribe._stars - 3);

    if(hasEnoughResources) {
        const bestMove = evaluateBestMove(state, (state: GameState) => {
            return generateTechMoves(state).filter(x => isTechWorthy(x.type));
        });
        // const bestMove2 = bestMove? evaluateBestMove(state, (state: GameState) => {
        //     return generateTechMoves(state).filter(x => x.id != bestMove.moves[0].id);
        // }) : null;
        
        if(bestMove && bestMove.score > evaluateEconomy(state)) {
            bestTechMove = bestMove.moves[0]!;
        }
    }

    // console.log(TechnologyType[Number(bestTechMove?.id.split('-')[1])]);
    
    // Do we have a good tech move to play?
    if(bestTechMove) {
        bestCityMoves['Technology'] = [bestTechMove];
    }

    return bestTechMove;
}

export function bestMoves(state: GameState, tryTech = true): BestMoves {
    const bestFinalMoves: BestMoves = { moves: [], score: 0 };
    const undoChain: UndoCallback[] = [];

    let economyMoves = bestEconomyMoves(state);
    let { scoreEconomy, starsLeft } = logAndUndoMoves(economyMoves?.moves || [], state, false, undoChain);
    
    if(economyMoves?.moves.length) {
        bestFinalMoves.moves.push(...economyMoves.moves);
    }

    const unitMoves = bestUnitMoves(state, economyMoves?.moves.length && starsLeft < 2? false : true);
    let { scoreArmy } = logAndUndoMoves(unitMoves, state, false, undoChain);
    if(unitMoves.length) {
        bestFinalMoves.moves.push(...unitMoves);
    }

    undoChain.reverse().forEach(x => x());

    if(tryTech) {
        const bestTechMove = generateBestTechMove(state);

        if(!bestTechMove) return bestFinalMoves;

        const undoTech = bestTechMove?.execute(state).undo;
        const moves = bestMoves(state, false);
        let { scoreEconomy: scoreEconomyTech, scoreArmy: scoreArmyTech } = logAndUndoMoves(moves.moves, state, false);
        undoTech();

        // If we didnt use whatever the tech unlocks, then it was no good
        if(!(() => { 
            const settings = TechnologySettings[bestTechMove.type as TechnologyType];
            for(const move of moves.moves) {
                switch (move.moveType) {
                    case MoveType.Summon:
                        if(settings.unlocksUnit && move.type == settings.unlocksUnit) {
                            return true;
                        }
                        break;
                    case MoveType.Build:
                        if(settings.unlocksStructure && move.type == settings.unlocksStructure) {
                            return true;
                        }
                        break;
                    case MoveType.Harvest:
                        if(settings.unlocksResource && move.type == settings.unlocksResource) {
                            return true;
                        }
                        break;
                }
            }
            return false;
        })()) {
            return bestFinalMoves;
        }
            
        const totalIncrease = scoreEconomyTech - scoreEconomy + scoreArmyTech - scoreArmy;

        if(totalIncrease > 0) {
            moves.moves.unshift(bestTechMove!);
            return moves;
        }
    }

    return bestFinalMoves;
}

/**
 * Returns the best set of economy-related moves for the current turn. 
 * It uses each city as reference point and looks ahead for the best combination of moves.
 * @param state
 * @param depth
 * @returns
 */
export function bestEconomyMoves(state: GameState, depth = 1, undoChain: UndoCallback[] = []): BestMoves | null {
    if(depth < 0) return null;

    const tribe = getPovTribe(state);
    let bestFinalMoves: BestMoves | null = null;

    // TODO use depth to evaluate turns-ahead 

    // Handle captures by just capturing them if possible
    // TODO or let army handle that?

    const villageCaptures = tribe._units.map(x => UnitMoveGenerator.captures(state, x)).flat().filter(x => 
        x && x.moveType == MoveType.Capture && state.tiles[x.src]._owner < 1
    ) as Move[];

    if(villageCaptures.length > 0) {
        // console.log('\nCaptures:');
        logAndUndoMoves(villageCaptures, state, false, undoChain);
        for(const capture of villageCaptures) {
            undoChain.push(capture.execute(state).undo);
        }
        bestFinalMoves = {
            moves: villageCaptures,
            score: evaluateEconomy(state)
        }
    } 

    const score = evaluateEconomy(state);

    // Ideally, we would choose the city with the best set of moves, then concat that with the second best.. etc
    // Sort by level up priority: Resources -> Workshop -> Super Unit -> Population Grouth

    const citiesByPriority: [BestMoves, number, number][] = [];

    for(let i = 0; i < tribe._cities.length; i++) {
        const city = tribe._cities[i];

        // How much population we have left to level up the city, and use that as depth
        // This helps keep strong moves first
        let remaining = (city._level + 1) - city._progress;

        // If its level 3 then we encourage finding more level up moves (lvl 3 rewards +3 pop)
        // Level 3 is important cause it can lead to getting a giant early
        if(city._level == 3) remaining += 4;

        const bestCityMoves = findBestMoves(state, remaining * 120, remaining, undefined, (state: GameState) => {
            const moves = [
                ...generateResourceMoves(state, city),
                ...generateStructureMoves(state, city),
                // ...(tribe._units.map(x => UnitMoveGenerator.captures(state, x)).filter(Boolean) as Move[]),
                // ...(state._scoreEconomy > 0? [generateEndTurnMove(state)] : []),
                // generateEndTurnMove(state),
            ];
            // if(moves.length < 2 && state._scoreEconomy < 1) return [];
            return moves;
        }, evaluateEconomy);

        if(!bestCityMoves || bestCityMoves.score < score) {
            // console.log(`No good moves for city ${city.name} (${bestCityMoves?.score || 0})`);
            continue;
        }

        const rewardType = bestCityMoves.moves.find(x => x.moveType == MoveType.Reward)?.type;

        switch (rewardType) {
            case RewardType.Explorer:
            case RewardType.Resources:
                citiesByPriority.push([bestCityMoves, 0, i]);
                break;
            case RewardType.Park:
            case RewardType.SuperUnit:
                citiesByPriority.push([bestCityMoves, 1, i]);
                break;
            case RewardType.CityWall:
            case RewardType.Workshop:
                citiesByPriority.push([bestCityMoves, 2, i]);
                break;
            case RewardType.BorderGrowth:
            case RewardType.PopulationGrowth:
                citiesByPriority.push([bestCityMoves, 3, i]);
                break;
            default:
                citiesByPriority.push([bestCityMoves, 4, i]);
                break;
        }
    }

    // Sort by priority
    const sortedCities = citiesByPriority.sort((a: any, b: any) => {
        if(b[1] > a[1]) return -1;
        return 0;
    });

    // Quick combine (may cause invalid moves?)
    let cityMoves: any[] = [];
    let combined: any = { };
    try {
        cityMoves = sortedCities.map(x => x[0].moves).flat();
        combined = logAndUndoMoves(cityMoves, state, false);
    } catch (error) {
        // TODO ?
        cityMoves = [];
        combined = {};
    }

    undoChain.reverse().forEach(x => x());
    
    return {
        moves: [
            ...bestFinalMoves? bestFinalMoves.moves : [], 
            ...cityMoves
        ],
        score: combined.scoreEconomy || 0
    }

    // Fallback to un-city bias
    
    // const bestEcoMoves = findBestMoves(state, 1000, 8, undefined, (state: GameState) => {
    //     if(state._scoreEconomy > 0) {
    //         if(hasCaptures) {
    //             return [
    //                 ...generateResourceMoves(state),
    //                 ...generateStructureMoves(state),
    //             ];
    //         }
    //         return tribe._units.map(x => UnitMoveGenerator.captures(state, x)).filter(Boolean) as Move[];
    //     }
    //     const moves = [
    //         ...generateResourceMoves(state),
    //         ...generateStructureMoves(state),
    //         // ...(tribe._units.map(x => UnitMoveGenerator.captures(state, x)).filter(Boolean) as Move[]),
    //         // ...(state._scoreEconomy > 0? [generateEndTurnMove(state)] : []),
    //         // generateEndTurnMove(state),
    //     ];
    //     if(moves.length < 2 && state._scoreEconomy < 1) return [];
    //     return moves;
    // }, evaluateEconomy);

    // // bestEcoMoves && logAndUndoMoves(bestEcoMoves.moves, state);

    // if(bestEcoMoves && bestEcoMoves.score > score) {
    //     bestFinalMoves = {
    //         moves: [
    //             ...(undoBestTechMove? [bestTechMove!] : []),
    //             ...bestEcoMoves.moves,
    //             // ...(!bestEcoMoves.moves[bestEcoMoves.moves.length-1].id.startsWith('end')? [generateEndTurnMove(state)] : []),
    //         ],
    //         score: bestEcoMoves.score
    //     }

    //     if(undoBestTechMove) {
    //         undoBestTechMove();
    //     }
    // }
    // else {
        
    //     if(undoBestTechMove) {
    //         undoBestTechMove();
    //     }

    //     const undo = generateEndTurnMove(state).execute(state).undo;

    //     score = evaluateEconomy(state);

    //     const bestEcoMoves: BestMoves | null = bestEconomyMoves(state, depth - 1);

    //     if(bestEcoMoves && bestEcoMoves.score > score) {
    //         bestFinalMoves = {
    //             moves: [
    //                 generateEndTurnMove(state),
    //                 ...bestEcoMoves.moves,
    //             ],
    //             score: bestEcoMoves.score
    //         }
    //     }

    //     undo();
    // }

    // // const moveIdxs = Object.values(bestPossibleMoves).map(x => x.map(y => y[1])).flat();
    // // const bestSequence = findBestMoves(state, 1000, 7, undefined, (state: GameState) => {
    // //     const bestSpawn = generateBestUnitSpawn(state);
    // //     return [
    // //         // ...(state.tribes[state.settings.pov]._units.map(x => UnitMoveGenerator.captures(state, x)).filter(Boolean) as Move[]),
    // //         ...generateStructureMoves(state),
    // //         ...(bestSpawn? [bestSpawn] : []),
    // //         // ...generateResourceMoves(state),
    // //         // generateEndTurnMove(state),
    // //     ]
    // //     // // Matches similar movetype and whatevertype
    // //     // ].filter(x => x.id.includes('reward') || moveIdxs.some(y =>
    // //     //     (y.toString().split('-')[0] == x.id.split('-')[0]
    // //     //     && y.toString().split('-')[1] == x.id.split('-')[1])
    // //     // ));
    // // }, evaluateEconomy);
    // // if(bestSequence && bestSequence.moves.length) bestCityMoves['Army'] = bestSequence.moves;

    // return bestFinalMoves;
}

function groupUnitsByProximity(state: GameState, tribe: TribeState, maxDistance: number = 4): UnitState[][] {
    const units = tribe._units;
    const groups: UnitState[][] = [];
    const unitToGroup: Map<UnitState, number> = new Map(); // Maps unit to its group index

    // Process each unit
    for (const unit of units) {
        let nearbyGroupIndices: Set<number> = new Set(); // Use Set to avoid duplicates

        // Check distance to all previously processed units
        for (const [otherUnit, groupIdx] of unitToGroup.entries()) {
            if (calculateDistance(unit._tileIndex, otherUnit._tileIndex, state.settings.size) <= maxDistance) {
                nearbyGroupIndices.add(groupIdx);
            }
        }

        if (nearbyGroupIndices.size === 0) {
            // No nearby units: create a new group
            groups.push([unit]);
            unitToGroup.set(unit, groups.length - 1);
        } else {
            // Nearby units found: merge into the primary group
            const primaryGroupIdx = Math.min(...nearbyGroupIndices); // Use the smallest index as the primary group

            // Add the current unit to the primary group
            groups[primaryGroupIdx].push(unit);
            unitToGroup.set(unit, primaryGroupIdx);

            // Merge other nearby groups into the primary group
            for (const groupIdx of nearbyGroupIndices) {
                if (groupIdx !== primaryGroupIdx) {
                    const unitsToMerge = groups[groupIdx];
                    groups[primaryGroupIdx].push(...unitsToMerge);
                    // Update group indices for all merged units
                    for (const mergedUnit of unitsToMerge) {
                        unitToGroup.set(mergedUnit, primaryGroupIdx);
                    }
                    // Clear the merged group
                    groups[groupIdx] = [];
                }
            }
        }
    }

    // Return only non-empty groups
    return groups.filter(group => group.length > 0);
}

export function bestUnitMoves(state: GameState, spawnsToo: boolean = false): Move[] {
    const tribe = getPovTribe(state);
    const unitGroups = groupUnitsByProximity(state, tribe, 2);
    const finalMoves: Move[] = [];
    
    for (const group of unitGroups) {
        const generator = (state: GameState) => {
            return group.map(unit => {
                const cityHome = tribe._cities.find(x => x.tileIndex == unit._homeIndex || x.tileIndex == state.tiles[unit._tileIndex]._rulingCityIndex);
                const bestSpawn = !spawnsToo? null : cityHome && generateBestUnitSpawn(state, cityHome);
                // const adjOwnedEmptyWaterTiles = !cityHome || tribe.stars < 7 || isTechLocked(tribe, TechnologyType.Fishing)? [] : getNeighborIndexes(state, unit._tileIndex).filter(x => state.tiles[x]._owner == tribe.owner && !state.structures[x] && isWaterTerrain(state.tiles[x]));
                return [
                    // Allow units to place a port to traverse on?
                    // ...adjOwnedEmptyWaterTiles.map(x => generateStructureMoves(state, cityHome, [x, StructureType.Port])).flat(),
                    // Generate all: steps, attacks and actions (capture, heal, etc)
                    ...UnitMoveGenerator.all(state, unit),
                    // Spawn a unit if possible
                    ...(bestSpawn? [bestSpawn] : []),
                    // Only allow to pass if we did something good
                    ...(state._potentialArmy > 1 || state._scoreTech > 1 ? [generateEndTurnMove()] : []),
                    // Add a pass move, to skip turn if all moves are bad!
                    // TODO
                ];
            }).flat();
        }

        // Find the best moves for this group
        const bestMoves = findBestMoves(state, 3000, 4 * group.length, .1, generator, evaluateArmy);

        if (bestMoves && bestMoves.moves.length) {
            for(const move of bestMoves.moves) {
                if(move.moveType == MoveType.EndTurn) break;
                finalMoves.push(move);
            }
        }
    }

    return finalMoves;
}