import { xorCity, xorUnit } from "../../zobrist/hasher";
import { attackUnit, removeUnit, summonUnit, gainStars, tryRemoveEffect, endUnitTurn } from "../actions";
import { getCityProduction, getEnemyAt, getPovTribe, getUnitAt, isSkilledIn, isTechUnlocked } from "../functions";
import Move, { CallbackResult, UndoCallback } from "../move";
import { GameState } from "../states";
import { EffectType, MoveType, SkillType, TechnologyType, TerrainType, UnitType } from "../types";

export default class Attack extends Move {
    constructor(src: number, target: number) {
        super(MoveType.Attack, src, target, null);
    }
    
    execute(state: GameState): CallbackResult {
        const attacker = getUnitAt(state, this.getSrc())!;
        
        // Units with infiltrate cannot attack units, instead they attack cities
        if(isSkilledIn(attacker, SkillType.Infiltrate)) {
            return this.riot(state);
        }
        
        const pov = getPovTribe(state);
        const defender = getEnemyAt(state, this.getTarget())!;
        const enemy = state.tribes[defender._owner];
        
        const result = attackUnit(state, attacker, defender);

        const undoChain: UndoCallback[] = [
            endUnitTurn(state, attacker)
        ];
        
        const undoEndTurn = endUnitTurn(state, attacker);
        let undoExtra = () => {};

        // allows a unit to convert an enemy unit into a friendly unit by attacking it
        // converted units take up population in the attacker's city but do not change score for either players
        if(isSkilledIn(attacker, SkillType.Convert)) {
            const index = enemy._units.findIndex(x => x._tileIndex == defender._tileIndex);
            
            xorUnit.owner(state, defender, enemy.owner, pov.owner);
            defender._owner = attacker._owner;
            enemy._units.splice(index, 1);
            
            undoChain.push(() => {
                xorUnit.owner(state, defender, pov.owner, enemy.owner);
                enemy._units.splice(index, 0, defender);
                defender._owner = enemy.owner;
            });
        }
        // normal attack, and attacker is still alive
        else if(attacker._health > 0) {
            // if unit was boosted, unboost
            undoChain.push(tryRemoveEffect(state, attacker, EffectType.Boost));

            // Units with Persist skill can keep on attacking
            if(defender._health <= 0 && isSkilledIn(attacker, SkillType.Persist)) {
                attacker._attacked = false;
                xorUnit.attacked(state, attacker);    

                undoChain.push(() => {
                    xorUnit.attacked(state, attacker);    
                    attacker._attacked = true;
                });
            }

            // Units with Escape skill can move after attacking
            if(attacker._health > 0 && isSkilledIn(attacker, SkillType.Escape)) {
                attacker._moved = false;
                xorUnit.moved(state, attacker);    

                undoChain.push(() => {
                    xorUnit.moved(state, attacker);    
                    attacker._moved = true;
                });
            }
        }

        return {
            rewards: result?.rewards || [],
            undo: () => {
                undoExtra();
                undoEndTurn();
                result?.undo();
            }
        };
    }
    
    riot(state: GameState): CallbackResult {
        const pov = getPovTribe(state);
        const infiltrator = getUnitAt(state, this.getSrc())!;
        const cityIndex = this.getTarget();
        const cityTile = state.tiles[cityIndex];
        const enemyTribe = state.tribes[cityTile._owner];
        const cityTarget = enemyTribe._cities.find(x => x.tileIndex == cityIndex)!;
        const enemyTarget = getUnitAt(state, cityIndex);
        
        // Cloak is consumed
        const undoConsume = removeUnit(state, infiltrator);

        let undoKillEnemy = () => {};
        
        // Any enemy unit in the city at the time will be damaged
        // This damage is equivalent to what a unit with an attack of 2 would deal
        if(enemyTarget) {
            // TODO not sure if they mean literally or by a calculated attack
            enemyTarget._health -= 2;

            if(enemyTarget._health <= 0) {
                const undoRemove = removeUnit(state, enemyTarget);
                pov._kills++;
                undoKillEnemy = () => {
                    pov._kills--;
                    undoRemove();
                    enemyTarget._health += 2;
                };
            }
            else {
                undoKillEnemy = () => {
                    enemyTarget._health += 2;
                };
            }
        }
        
        // A group of Daggers will spawn in the city's tile. 
        // Daggers will prioritize spawning on terrain whey they can benefit from a defense bonus.
        let defTiles: number[] = [];
        let waterTiles: number[] = [];
        let otherTiles: number[] = [];
        
        cityTarget._territory.forEach(x => {
            if(state.tiles[x]._unitOwner > 0 || x == cityIndex) {
                return;
            }
            switch (state.tiles[x].terrainType) {
                case TerrainType.Mountain:
                    if(isTechUnlocked(pov, TechnologyType.Climbing)) {
                        defTiles.push(x);
                    }
                    break;
                case TerrainType.Forest:
                    if(isTechUnlocked(pov, TechnologyType.Archery)) {
                        defTiles.push(x);
                    }
                    break;
                case TerrainType.Water:
                    if(isTechUnlocked(pov, TechnologyType.Sailing)) {
                        waterTiles.push(x);
                    }
                    break;
                default:
                    otherTiles.push(x);
                    break;
            }
        });
        
        otherTiles.sort(() => Math.random() - 0.5);
        defTiles.sort(() => Math.random() - 0.5);
        waterTiles.sort(() => Math.random() - 0.5);
        
        // If the enemy died or there wasnt any, then a dagger spawns on the city tile
        if(enemyTarget && enemyTarget._health < 0) {
            // Move to front, guarentee the unit spawns there first
            if(otherTiles.includes(cityIndex)) {
                otherTiles.splice(otherTiles.indexOf(cityIndex), 1);
                otherTiles = [cityIndex, ...otherTiles];
            }
        }
        
        // They will not be able to perform any actions until the next turn.
        // On water tiles, pirates spawn in stead of daggers
        // A Dagger will only spawn as a Pirate if there are no empty land tiles remaining within the borders of an infiltrated city. 
        // The number of daggers is relative to the city's size, with a max of 5 daggers.
        
        const daggers: UndoCallback[] = [];
        const rewards = [];
        const income = Math.min(5, getCityProduction(state, cityTarget));

        for (let _ = 0; _ < income; _++) {
            let tileIndex = defTiles.pop() || otherTiles.pop();
            let unitType = UnitType.Dagger;
            if(!tileIndex)  {
                // water tile
                tileIndex = waterTiles.pop();
                if(!tileIndex) break;
                unitType = UnitType.Pirate;
            } 
            const result = summonUnit(state, unitType, tileIndex);
            if(result) {
                rewards.push(...result.rewards);
                daggers.push(result.undo);
            }
        }
        
        // The infiltrating player will immediately gain a number of stars equal to the income of the city.
        const undoStars = gainStars(state, income);

        // The city will produce zero stars on their opponent's next turn. 
        // but will not affect other methods of star production. (eg: markets, diplomacy)
        
        xorCity.riot(state, cityTarget);
        cityTarget._riot = true;
        
        return {
            rewards: rewards,
            undo: () => {
                xorCity.riot(state, cityTarget);
                cityTarget._riot = false;
                undoStars();
                daggers.forEach(x => x());
                undoKillEnemy();
                undoConsume();
            }
        };
    }
}