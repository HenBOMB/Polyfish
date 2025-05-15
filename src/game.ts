import { GameState } from "./core/states";
import { STARTING_OWNER_ID } from "./core/gameloader";
import { MoveGenerator } from "./core/moves";
import { cloneState, getCityProduction, getPovTribe, hasEffect, isGameOver } from "./core/functions";
import { tryDiscoverRewardOtherTribes } from "./core/actions";
import Move, {UndoCallback } from "./core/move";
import { MoveType } from "./core/types";
import { EffectType } from "./core/types";
import NetworkManager from "./core/network";
import PoseManager from "./core/poser";
import { endUnitTurn, gainStars, startUnitTurn, tryRemoveEffect } from "./core/actions";

export default class Game {
    initialState: GameState;
    state: GameState;
    network: NetworkManager;
    poser: PoseManager;

    constructor() {
        this.initialState = {} as any;
        this.state = {} as any;
        this.network = null as any;
        this.poser = new PoseManager();
    }

    deepClone() {
        const game = new Game();
        game.load(this.initialState);
        return game;
    }

    cloneState() {
        return cloneState(this.state);
    }

    load(state: GameState) {
        this.initialState = state;
        this.reset();
    }

    reset() {
        this.state = this.initialState;
        // this.state = cloneState(this.initialState);
        this.network = new NetworkManager(this.state);
        this.state.tiles.forEach(tile => {
            this.state._visibleTiles[tile.tileIndex] = tile._explorers.has(this.state.settings._pov);
        });
    }

    // TODO XOR?

    playMove(moveOrIndex: number | Move): [Move, UndoCallback] | null {
        if(moveOrIndex === null || moveOrIndex === undefined) {
            throw new Error('Move is undefiend!');
        }
        if(this.state.settings._gameOver) {
            return null;
        }

        const move = typeof moveOrIndex == "number" ? MoveGenerator.legal(this.state)[moveOrIndex] : moveOrIndex;
        let undo: UndoCallback;

        this.state.settings.areYouSure = true;

        if(move.moveType === MoveType.EndTurn) {
            undo = this.endTurn();
            this.state.settings._recentMoves = [];
        }
        else {
		    const result = move.execute(this.state)!;
            const undoDiscover = tryDiscoverRewardOtherTribes(this.state);

            // TODO also need a function to update the diplomacy vision of the discovered tribes
            // if researched tech
            
            undo = () => {
                undoDiscover();
                result.undo();
            }

            // If we just played a reward move, clear the first two
            if(move.moveType == MoveType.Reward) {
                this.state.settings._pendingRewards.splice(0, 2);
            }

            // If playing the move lead to rewards, queue them
            if(result.rewards) {
                this.state.settings._pendingRewards.push(...result.rewards);
            }
        }


        this.state.settings.areYouSure = false;

        return [move, undo];
    }
    
    // TODO XOR

    /**
     * Ends the current tribe's turn
     */
    endTurn(): UndoCallback {
        const state = this.state;

        // TODO Add relations? (for polytopia default bots)
        const chain: UndoCallback[] = [];

        // ! CURRENT TURN ! //

        const oldpov = state.settings._pov;
        const oldTurn = state.settings._turn;

        chain.unshift(() => {
            state.settings._pov = oldpov;
            state.settings._turn = oldTurn;
        });

        state.settings._pov++;
        if(state.settings._pov > state.settings.tribeCount) {
            state.settings._pov = STARTING_OWNER_ID;
        }
        let pov = getPovTribe(state);

        // Search for the next tribe
        while(pov._killedTurn > 0 || pov._resignedTurn > 0) {
            state.settings._pov++;
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
                chain.forEach(x => x());
            }
        }

        // ! NEW TRIBE TURN ! //
    
        // Update the new tribe's visibility
        const oldVisibility = { ...state._visibleTiles };
        
        state.tiles.forEach(tile => {
            state._visibleTiles[tile.tileIndex] = tile._explorers.has(pov.owner);
        });

        chain.unshift(() => {
            state._visibleTiles = oldVisibility;
        });

        // Trigger disovery if some other tribes moved into our visible terrain
        chain.unshift(tryDiscoverRewardOtherTribes(state));

        // TODO also need a function to update the diplomacy vision of the discovered tribes

        // Reward tribe with its production if its not the first turn
        if(state.settings._turn > 1) {
            chain.unshift(gainStars(state, getCityProduction(state, ...pov._cities)));
        }

        // Update all unit states
		for (let i = 0; i < pov._units.length; i++) {
			const unit = pov._units[i];

            // Frozen units get unfrozen but that end their turn
			if(hasEffect(unit, EffectType.Frozen)) {
                chain.unshift(tryRemoveEffect(state, unit, EffectType.Frozen));
                chain.unshift(endUnitTurn(state, unit));
			}
            else {
                chain.unshift(startUnitTurn(state, unit));
            }
		}

        return () => {
            chain.forEach(x => x());
        }
    }
}