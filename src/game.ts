import { GameState } from "./core/states";
import { STARTING_OWNER_ID } from "./core/gameloader";
import { MoveGenerator } from "./core/moves";
import { cloneState, getCityProduction, getPovTribe, isFrozen, isGameOver, tryDiscoverRewardOtherTribes } from "./core/functions";
import Move, {UndoCallback } from "./core/move";
import { MoveType } from "./core/types";
import { EffectType } from "./core/types";
import NetworkManager from "./core/network";

export default class Game {
    initialState: GameState;
    state: GameState;
    network: NetworkManager;

    constructor() {
        this.initialState = {} as any;
        this.state = {} as any;
        this.network = null as any;
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
        this.state = cloneState(this.initialState);
        this.network = new NetworkManager(this.state);
    }

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
            // TODO how about updating the state in here? to compare start to end of current turn?
            undo = this.endTurn();
            this.state.settings._recentMoves = [];
        }
        else {
		    const result = move.execute(this.state)!;
            undo = result.undo;

            // If we just played a reward move, clear the first two
            if(move.moveType == MoveType.Reward) {
                this.state.settings._pendingRewards.splice(0, 2);
            }

            // If playing the move lead to rewards, queue them
            if(result.rewards) {
                this.state.settings._pendingRewards.push(...result.rewards);
            }
        }

		const undoDiscover = tryDiscoverRewardOtherTribes(this.state);

        this.state.settings.areYouSure = false;

        return [move, () => {
            undoDiscover();
            undo();
        }];
    }
    
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