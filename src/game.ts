import { GameState } from "./core/states";
import AIState from "./aistate";
import { STARTING_OWNER_ID } from "./core/gameloader";
import { MoveGenerator, safelyExecuteMove } from "./core/moves";
import { cloneState, getCityProduction, getNeighborIndexes, getPovTribe, isFrozen, isGameOver, tryDiscoverRewardOtherTribes } from "./core/functions";
import Move, { Branch, CallbackResult, UndoCallback } from "./core/move";
import { MoveType } from "./core/types";
import { EffectType } from "./core/types";

export default class Game {
    initialState: GameState;
    stateBefore: GameState;
    state: GameState;

    constructor() {
        this.initialState = {} as any;
        this.state = {} as any;
        this.stateBefore = {} as any;
    }

    deepClone() {
        const game = new Game();
        game.load(this.initialState);
        return game;
    }

    load(state: GameState) {
        this.initialState = state;
        this.reset();
    }

    reset() {
        this.stateBefore = cloneState(this.initialState);
        this.state = cloneState(this.initialState);
    }

    playMove(moveOrIndex: number | Move): [Move, UndoCallback] | null {
        if(this.state.settings._gameOver) {
            return null;
        }

        this.stateBefore = cloneState(this.state);

        const move = typeof moveOrIndex == "number" ? MoveGenerator.legal(this.state)[moveOrIndex] : moveOrIndex;
        let undo: UndoCallback;

        this.state.settings.areYouSure = true;

        if(move.moveType === MoveType.EndTurn) {
            // TODO how about updating the state in here? to compare start to end of current turn?
            undo = this.endTurn();
            this.state.settings._recentMoves = [];
        }
        else {
            undo = safelyExecuteMove(this.state, move) || (() => {});
        }

        this.state.settings.areYouSure = false;

        return [move, undo];
    }
    
    // /**
    //  * Generates random moves by selecting random moves and shuffling them
    //  */
    // getGoodMoves(randomChance = 0.7): Move[] {
    //     const actual = cloneState(this.state);
    //     const doRandom = Math.random() < randomChance;

    //     const maxMoves = 7;
    //     const maxDepth = doRandom? 1 : 2;
    //     const maxArmyMoves = 10;
    //     const maxArmyDepth = doRandom? 1 : 2;

    //     // Use actual state to generate moves, or state will not match

    //     // const ecoMoves: BestMoves = null as any;
    //     const ecoMoves = evaluateBestMove(
    //         this.state, 
    //         (state: GameState) => generateEcoMoves(state)
    //             .sort(() => 0.5 - Math.random())
    //             // .slice(minMoves, minMoves + Math.floor(Math.random() * (maxMoves - minMoves + 1)))
    //             .slice(0, maxMoves)
    //             .filter(Boolean), 
    //             doRandom? () => Math.random() * 10 : evaluateEconomy, 
    //         maxDepth
    //     );

    //     if(!deepCompare(actual, this.state, 'state', true)) {
    //         throw Error('ECO COMPARE');
    //     }

    //     const undoChain: UndoCallback[] = [];

    //     if(ecoMoves?.moves.length) {
    //         logAndUndoMoves(ecoMoves.moves, this.state, false, undoChain);
    //     }

    //     const unitMoves = evaluateBestMove(
    //         this.state, 
    //         (state: GameState) => [
    //             ...getPovTribe(state)._units.map(x => UnitMoveGenerator.all(state, x)).flat(),
    //             ...UnitMoveGenerator.spawns(state),
    //         ].sort(() => 0.5 - Math.random()).slice(0, maxArmyMoves).filter(Boolean), 
    //         doRandom? () => Math.random() * 10 : evaluateArmy, 
    //         maxArmyDepth
    //     );

    //     undoChain.reverse().forEach(x => x());

    //     if(!deepCompare(actual, this.state, 'state', true)) {
    //         throw Error('ARMY COMPARE');
    //     }

    //     // if(ecoMoves) {
    //     //     const start = ecoMoves.moves.findIndex(x => x.id.startsWith('end'));
    //     //     ecoMoves.moves = ecoMoves.moves.slice(0, start);
    //     // }

    //     // if(unitMoves) {
    //     //     const start = unitMoves.moves.findIndex(x => x.id.startsWith('end'));
    //     //     unitMoves.moves = unitMoves.moves.slice(0, start);
    //     // }
        
    //     return [
    //         ...ecoMoves?.moves || [],
    //         ...unitMoves?.moves || [],
    //     ];
    // }
    
    /**
     * Ends the current tribe's turn
     */
    endTurn(): UndoCallback {
        const state = this.state;

        // TODO Add relations? (for polytopia default bots)
        const chain: UndoCallback[] = [];

        // Cycles turns without overflow
        const nextPov = () => {
            // Continue with next tribe
            state.settings._pov += 1;
            // If overflowing, go back to start
            if(state.settings._pov > state.settings.tribeCount) {
                state.settings._pov = STARTING_OWNER_ID;
            }
        }
        
        const oldpov = state.settings._pov;
        const oldTurn = state.settings._turn;

        chain.push(() => {
            state.settings._pov = oldpov;
            state.settings._turn = oldTurn;
        });

        nextPov();

        let pov = getPovTribe(state);

        // Search for the next tribe
        while(pov._killedTurn > 0 || pov._resignedTurn > 0) {
            nextPov();
            pov = getPovTribe(state);
        }

        // If we are back at the start, a new turn has started
        if(state.settings._pov === STARTING_OWNER_ID) {
            state.settings._turn++;
        }

        // Dont continue if the game has ended
        if(isGameOver(state)) {
            state.settings._gameOver = true;
            return () => {
                state.settings._gameOver = false;
                chain.reverse().forEach(x => x());
            }
        }

        // New tribe POV

        const oldStars = pov._stars;

        // Reward tribe with its production if its not the first turn
        if(state.settings._turn > 1) {
            pov._stars += getCityProduction(state, ...pov._cities);
        }
        
        chain.push(() => {
            pov._stars = oldStars;
        });

        // Update all unit states
		for (let i = 0; i < pov._units.length; i++) {
			const unit = pov._units[i];
            const moved = unit._moved;
            const attacked = unit._attacked;

            chain.push(() => {
                unit._moved = moved;
                unit._attacked = attacked;
            });

            // Frozen units get unfrozen but that consumes their turn
            // not in wiki but i assume this is how it works from gameplay
			if(isFrozen(unit)) {
                const index = unit._effects.indexOf(EffectType.Frozen);
				unit._effects.splice(index, 1);
                chain.push(() => {
                    unit._effects.splice(index, 0, EffectType.Frozen);
                });
				unit._moved = unit._attacked = true;
				continue;
			}

			unit._moved = unit._attacked = false;
		}

        // Trigger disovery if some other tribes moved into our visible terrain
        chain.push(tryDiscoverRewardOtherTribes(state));

        // Update the new tribe's visibility
        const visible = [...state._visibleTiles];
        const lighthouses = [...state._lighthouses];

        chain.push(() => {
            state._visibleTiles = visible;
            state._lighthouses = lighthouses;
        });

        state._visibleTiles = [];

        Object.values(state.tiles).forEach(tile => {
            if(tile._explorers.includes(pov.owner)) {
                state._visibleTiles.push(tile.tileIndex);
            }
        });

        state._lighthouses = [
            0,
            state.settings.size - 1,
            state.settings.size * state.settings.size - 1,
            1 + state.settings.size * state.settings.size - state.settings.size
        ].filter(x => !state.tiles[x]._explorers.includes(pov.owner));

        return () => {
            chain.reverse().forEach(x => x());
        }
    }
}