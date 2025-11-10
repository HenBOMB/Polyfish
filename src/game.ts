import { Coords, GameSettings, GameState, PartialGameSettings, TileState, TribeState } from "./core/states";
import GameLoader, { STARTING_OWNER_ID } from "./core/gameloader";
import { MoveGenerator } from "./core/moves";
import { _setVisibleTiles, getCityProduction, getPovTribe, hasEffect, isGameOver } from "./core/functions";
import { tryDiscoverRewardOtherTribes } from "./core/actions";
import Move, {UndoCallback } from "./core/move";
import { MoveType } from "./core/types";
import { EffectType } from "./core/types";
import NetworkManager from "./core/network";
import PoseManager from "./core/poser";
import { endUnitTurn, gainStars, startUnitTurn, tryRemoveEffect } from "./core/actions";
import UnitValues from "./ai/values/Units";
import TechnologyValues from "./ai/values/Technology";
import CapturePotentialValues from "./ai/values/CapturePotential";

type RegisteredValues = {
    unitStrength: UnitValues,
    technology: TechnologyValues,
    capturePotential: CapturePotentialValues,
};

export default class Game {
    state: GameState;
    // network: NetworkManager;
    // poser: PoseManager;
    values: RegisteredValues;

    constructor() {
        this.state = { } as any;
        this.values = { } as any;
        // this.network = null as any;
        // this.poser = new PoseManager();
        // this.poser = null as any;
        // console.log(`[Game] Poser disabled`);
    }

    clone() {
        const clone = new Game();
        clone.load(this.state);
        const pov = getPovTribe(clone.state);
        clone.values = {
            unitStrength: new UnitValues(pov.type),
            technology: new TechnologyValues(pov.type),
            capturePotential: new CapturePotentialValues(pov.type),
        }
        return clone;
    }

    cloneState(): GameState {
        // Copied logic from cloneState
        // const hashes: string[] = [];
        // for (let i = 0; i < hashes.length; i++) {
        //     this.state.tribes[i + 1].hash = BigInt(hashes[i]);
        // }
        const state = Game.deserializeState(Game.serializeState(this.state));
        return state;
    }

    static serializeState(state: GameState): string {
        // BigInt and Set handled by index.ts so no nothing else is needed
        const serialized = JSON.stringify({
            ...state,
            settings: {
                ...state.settings,
                _pendingRewards: state.settings._pendingRewards.map(x => Move.serialize(x))
            }
        });
        return serialized;
    }

    static deserializeState(json: string): GameState {
        const parseCoords = (coords: { x: number, y: number, idx: number }) => Coords.from(coords.x, coords.y, state);
        const state = JSON.parse(json) as GameState;
        const rebuilt = {
            _visibleTiles: { ...state._visibleTiles },
            resources:     { ...state.resources },
            structures:    { ...state.structures },
            settings:      { 
                ...state.settings,
                _pendingRewards: MoveGenerator.fromActions(
                    state.settings._pendingRewards.map(x => Move.deserialize(x as any))
                )
            },
            tiles: Object.keys(state.tiles).reduce((acc, idx) => {
                const tile = state.tiles[Number(idx)];
                return { 
                    ...acc, 
                    [idx]: {
                        ...tile,
                        coords: parseCoords(tile.coords),
                        rulingCityCoords: tile.rulingCityCoords? parseCoords(tile.rulingCityCoords) : undefined,
                        explorers: new Set(Object.values(tile.explorers)),
                    } as TileState
                };
            }, { }),
            tribes: Object.values(state.tribes).reduce((acc, tribe) => ({ 
                ...acc,
                [tribe.id]: {   
                    ...tribe,
                    _hash: BigInt(tribe._hash.toString()),
                    tech_vanilla: tribe.tech_vanilla.map(x => ({ 
                        ...x 
                    })),
                    builtUniqueImprovements: new Set(Object.values(tribe.builtUniqueImprovements)),
                    knownPlayers: new Set(Object.values(tribe.knownPlayers)),
                    cities: tribe.cities.map(x => ({ 
                        ...x, 
                        rewards: new Set(Object.values(x.rewards))
                    })),
                    units: tribe.units.map(unit => ({ 
                        ...unit, 
                        effects: new Set(Object.values(unit.effects)),
                        coords: parseCoords(unit.coords),
                        homeCoords: unit.homeCoords && unit.homeCoords?.x != -1? parseCoords(unit.homeCoords!) : undefined,
                        prevCoords: parseCoords(unit.prevCoords),
                    })),
                    relations: Object.entries(tribe.relations).reduce((x, [k, v]) => ({ 
                        ...x, [k]: { ...v } 
                    }), {})
                } as TribeState
            }), {}),
        };
        (rebuilt as any)._hiddenResources = (state as any)._hiddenResources;
        return rebuilt;
    }

    static serialize(game: Game): string {
        return JSON.stringify({
            stateJson: Game.serializeState(game.state)
        });
    }

    static deserialize(json: string): Game {
        const game = new Game();
        const data = JSON.parse(json) as { stateJson: string };
        const state = Game.deserializeState(data.stateJson);
        game.load(state);
        const pov = getPovTribe(state);
        game.values = {
            unitStrength: new UnitValues(pov.type),
            technology: new TechnologyValues(pov.type),
            capturePotential: new CapturePotentialValues(pov.type),
        }
        return game;
    }

    public async loadRandom(settings?: PartialGameSettings) {
        const loader = new GameLoader();
        await loader.loadRandom(settings);
        this.load(loader.currentState);
    }

    public async loadLive(settings?: PartialGameSettings) {
        const loader = new GameLoader();
        await loader.loadLive(settings).catch((err) => {
            console.log('Failed to load live, trying random');
            console.log(err);
            return loader.loadRandom();
        });
        this.load(loader.currentState);
    }

    public load(state: GameState) {
        // this.initialState = state;
        // this.network = new NetworkManager(this.state);
        // this.reset();
        this.state = state;
        this.state = this.cloneState();
        // this.network = new NetworkManager(this.state);
        _setVisibleTiles(this.state, this.state.settings.currentPlayerTurnId);
        const pov = getPovTribe(state);
        this.values = {
            unitStrength: new UnitValues(pov.type),
            technology: new TechnologyValues(pov.type),
            capturePotential: new CapturePotentialValues(pov.type),
        }
    }

    playSequence(...legalMoveIndexes: number[]): UndoCallback[] {
        const undos = [];
        for (let i = 0; i < legalMoveIndexes.length; i++) {
            const id = legalMoveIndexes[i];
            const move = MoveGenerator.legal(this.state)[id];
            const moveData = this.playMove(move);
            if (!moveData) {
                throw new Error('Invalid move');
            }
            undos.unshift(moveData[1]);
        }
        return undos;
    }

    // TODO XOR?

    playMove(moveOrIndex: number | Move): [Move, UndoCallback] | null {
        if(moveOrIndex === null || moveOrIndex === undefined) {
            throw new Error('Move is undefiend!');
        }
        if(this.state.settings._gameOver) {
            return null;
        }

        const legal = MoveGenerator.legal(this.state);
        const move = typeof moveOrIndex == "number" ? legal[moveOrIndex] : moveOrIndex;

        if(!move) {
            console.log(moveOrIndex, legal.length);
            throw new Error('Move is undefiend!');
        }
        // console.log(`[Game] legal: ${legal.length}`);
        // console.log(`[Game] playing ${moveOrIndex} -> ${move?.stringify(this.state, this.state) || '???'}`);
        
        let undo: UndoCallback;

        this.state.settings._areYouSure = true;

        if(move.moveType === MoveType.EndTurn) {
            undo = this.endTurn();
            this.state.settings._recentMoves = [];
        }
        else {
		    const result = move.execute(this.state);
            const undoDiscover = tryDiscoverRewardOtherTribes(this.state);

            // TODO also need a function to update the diplomacy vision of the discovered tribes
            // if researched tech

            const oldRewards = [...this.state.settings._pendingRewards];

            // If we just played a reward move, clear the first two
            if(move.moveType == MoveType.Reward) {
                this.state.settings._pendingRewards.splice(0, 2);
            }

            // If playing the move lead to rewards, queue them
            if(result.rewards) {
                this.state.settings._pendingRewards.push(...result.rewards);
            }

            this.state.settings._recentMoves.push(move.moveType);

            undo = () => {
                this.state.settings._recentMoves.pop()
                this.state.settings._pendingRewards = oldRewards;
                undoDiscover();
                result.undo();
            }
        }

        this.state.settings._areYouSure = false;

        return [move, undo];
    }
    
    // TODO XOR

    /**
     * Ends the current tribe's turn
     */
    private endTurn(): UndoCallback {
        const state = this.state;
        const oldpov = state.settings.currentPlayerTurnId;
        const oldTurn = state.settings.turn;

        // TODO Add relations? (for polytopia default bots)
        const chain: UndoCallback[] = [
            () => {
                state.settings.currentPlayerTurnId = oldpov;
                state.settings.turn = oldTurn;
            }
        ];

        let pov = getPovTribe(state);

        // ! CURRENT TURN ! //

        // TODO units auto-recover if they didnt use up any of their moves
        
        // ! CHANGE TURN ! //
        const lastPov = state.settings._lastPlayerTurnId;
        state.settings._lastPlayerTurnId = state.settings.currentPlayerTurnId;
        state.settings.currentPlayerTurnId++;
        if(state.settings.currentPlayerTurnId > state.settings._maxTribeCount) {
            state.settings.currentPlayerTurnId = STARTING_OWNER_ID;
        }
        pov = getPovTribe(state);

        // Search for the next tribe
        while(pov.killedTurn > 0 || pov.resignedTurn > 0) {
            state.settings.currentPlayerTurnId++;
            pov = getPovTribe(state);
        }

        // If we are back at the start, a new turn has started
        if(state.settings.currentPlayerTurnId === STARTING_OWNER_ID) {
            state.settings.turn++;
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
        
        _setVisibleTiles(state, pov.id);

        chain.unshift(() => {
            state._visibleTiles = oldVisibility;
        });

        // Trigger disovery if some other tribes moved into our visible terrain
        chain.unshift(tryDiscoverRewardOtherTribes(state));

        // TODO also need a function to update the diplomacy vision of the discovered tribes

        // Reward tribe with its production if its not the first turn
        if(state.settings.turn > 1) {
            const spt = getCityProduction(state, ...pov.cities);
            if (spt > 0) {
                chain.unshift(gainStars(state, spt));
            }
        }

        // Update all unit states
		for (let i = 0; i < pov.units.length; i++) {
			const unit = pov.units[i];

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
            state.settings._lastPlayerTurnId = lastPov;
        }
    }

    /**
     * Executes a move in a provided GameState
     * @param state 
     * @param move 
     * @returns 
     */
    static playMove(state: GameState, move: Move): [Move, UndoCallback] | null {
        if(!move) {
            throw new Error('Move was not assigned!');
        }

        if(state.settings._gameOver) {
            return null;
        }

        // console.log(`[Game] legal: ${legal.length}`);
        // console.log(`[Game] playing ${moveOrIndex} -> ${move?.stringify(this.state, this.state) || '???'}`);
        
        let undo: UndoCallback;

        state.settings._areYouSure = true;

        if(move.moveType === MoveType.EndTurn) {
            undo = this.endTurn(state);
            state.settings._recentMoves = [];
        }
        else {
		    const result = move.execute(state)!;
            const undoDiscover = tryDiscoverRewardOtherTribes(state);

            // TODO also need a function to update the diplomacy vision of the discovered tribes
            // if researched tech

            const oldRewards = [...state.settings._pendingRewards];

            // If we just played a reward move, clear the first two
            if(move.moveType == MoveType.Reward) {
                state.settings._pendingRewards.splice(0, 2);
            }

            // If playing the move lead to rewards, queue them
            if(result.rewards) {
                state.settings._pendingRewards.push(...result.rewards);
            }

            state.settings._recentMoves.push(move.moveType);

            undo = () => {
                state.settings._recentMoves.pop()
                state.settings._pendingRewards = oldRewards;
                undoDiscover();
                result.undo();
            }
        }

        state.settings._areYouSure = false;

        return [move, undo];
    }

    /**
     * Executes ending the turn in a provided GameState
     * @param state 
     * @returns 
     */
    static endTurn(state: GameState): UndoCallback {
        const oldpov = state.settings.currentPlayerTurnId;
        const oldTurn = state.settings.turn;

        // TODO Add relations? (for polytopia default bots)
        const chain: UndoCallback[] = [
            () => {
                state.settings.currentPlayerTurnId = oldpov;
                state.settings.turn = oldTurn;
            }
        ];

        let pov = getPovTribe(state);

        // ! CURRENT TURN ! //

        // TODO units auto-recover if they didnt use up any of their moves
        
        // ! CHANGE TURN ! //

        state.settings.currentPlayerTurnId++;
        if(state.settings.currentPlayerTurnId > state.settings._maxTribeCount) {
            state.settings.currentPlayerTurnId = STARTING_OWNER_ID;
        }
        pov = getPovTribe(state);

        // Search for the next tribe
        while(pov.killedTurn > 0 || pov.resignedTurn > 0) {
            state.settings.currentPlayerTurnId++;
            pov = getPovTribe(state);
        }

        // If we are back at the start, a new turn has started
        if(state.settings.currentPlayerTurnId === STARTING_OWNER_ID) {
            state.settings.turn++;
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
        
        _setVisibleTiles(state, pov.id);

        chain.unshift(() => {
            state._visibleTiles = oldVisibility;
        });

        // Trigger disovery if some other tribes moved into our visible terrain
        chain.unshift(tryDiscoverRewardOtherTribes(state));

        // TODO also need a function to update the diplomacy vision of the discovered tribes

        // Reward tribe with its production if its not the first turn
        if(state.settings.turn > 1) {
            chain.unshift(gainStars(state, getCityProduction(state, ...pov.cities)));
        }

        // Update all unit states
		for (let i = 0; i < pov.units.length; i++) {
			const unit = pov.units[i];

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

