import { xorCity, xorUnit } from "../../zorbist/hasher";
import { attackUnit, removeUnit, summonUnit, gainStars, tryRemoveEffect } from "../actions";
import { getEnemyAt, getPovTribe, getUnitAt, hasEffect, isSkilledIn, isTechUnlocked } from "../functions";
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
        if (isSkilledIn(attacker, SkillType.Infiltrate)) {
            return this.riot(state);
        }
        
        const pov = getPovTribe(state);
        const defender = getEnemyAt(state, this.getTarget())!;
        const moved = attacker._moved;
        const enemy = state.tribes[defender._owner];
        
        let undoAttack = () => {};
        let undoExtra = () => {};

        // Allows a unit to convert an enemy unit into a friendly unit by attacking it
        // TODO verify if working: Converted units take up population in the attacker's city but do not change score for either players
        if(isSkilledIn(attacker, SkillType.Convert)) {
            const owner = defender._owner;
            const index = enemy._units.findIndex(x => x._tileIndex == defender._tileIndex);
            
            xorUnit.owner(state, defender, enemy.owner, pov.owner);
            defender._owner = attacker._owner;
            enemy._units.splice(index, 1);
            attacker._attacked = attacker._moved = true;
            
            undoAttack = () => {
                xorUnit.owner(state, defender, pov.owner, enemy.owner);
                attacker._attacked = false;
                attacker._moved = moved;
                enemy._units.splice(index, 0, defender);
                defender._owner = owner;
            }
        }
        else {
            undoAttack = attackUnit(state, attacker, defender);
            
            // If defender was not killed 
            if(defender._health > 0) {
                
            }
            // Units with Persist skill can keep on killing if they one shot the defender
            else if(attacker._health > 0 && isSkilledIn(attacker, SkillType.Persist)) {
                attacker._attacked = false;
            }
            else {
                attacker._attacked = true;
            }

            // attacker is still alive
            if(attacker._health > 0) {
                // if unit was boosted, unboost
                undoExtra = tryRemoveEffect(state, attacker, EffectType.Boost);
            }
            
            // Units with Escape can move after they attacked
            if (attacker._health > 0 && isSkilledIn(attacker, SkillType.Escape)) {
                attacker._moved = false;
            }
            else {
                attacker._moved = true;
            }
        }
        
        return {
            rewards: [],
            undo: () => {
                attacker._moved = moved;
                attacker._attacked = false;
                undoExtra();
                undoAttack();
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
        
        // Cloak is consumed
        const undoConsume = removeUnit(state, infiltrator);
        
        // Any enemy unit in the city at the time will be damaged
        const enemyTarget = getUnitAt(state, cityIndex);
        
        let undoKillEnemy = () => {};
        
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
            const tile = state.tiles[x];
            if(tile._unitOwner > 0 || x == cityIndex) return;
            switch (tile.terrainType) {
                case TerrainType.Mountain:
                if(isTechUnlocked(pov, TechnologyType.Climbing)) {
                    defTiles.push(x);
                }
                return
                case TerrainType.Forest:
                if(isTechUnlocked(pov, TechnologyType.Archery)) {
                    defTiles.push(x);
                }
                return
                case TerrainType.Water:
                if(isTechUnlocked(pov, TechnologyType.Sailing)) {
                    waterTiles.push(x);
                }
                return
            }
            otherTiles.push(x);
            return;
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

        for (let j = 0; j < Math.min(5, cityTarget._production); j++) {
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
        // TODO re-verify: THIS IS INCORRECT, I TESTED IT AND ITS THE AMOUNT OF DAGGERS SPAWNED!
        
        const undoStars = gainStars(state, daggers.length);

        // The city will produce zero stars on their opponent's next turn. 
        // but will not affect other methods of star production. (eg: markets, diplomacy)
        
        xorCity.riot(state, cityTarget, false);
        cityTarget._riot = true;
        
        return {
            rewards: rewards,
            undo: () => {
                xorCity.riot(state, cityTarget, true);
                cityTarget._riot = false;
                undoStars();
                daggers.forEach(x => x());
                undoKillEnemy();
                undoConsume();
            }
        };
    }
}