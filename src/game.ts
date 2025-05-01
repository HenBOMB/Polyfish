import { GameState, TribeState } from "./core/states";
import AIState from "./aistate";
import { STARTING_PLAYER } from "./core/gameloader";
import UnitMoveGenerator, { generateAllMoves, generateEcoMoves, generateEndTurnMove, UndoCallback } from "./core/moves";
import { cloneState, getNeighborIndexes, getPovTribe, getStarExchange, getUnitAtTile, isGameOver, isGameWon } from "./core/functions";
import { evaluateArmy, evaluateEconomy } from "./eval/eval";
import { evaluateBestMove, logAndUndoMoves } from "./eval/evalMoves";
import { deepCompare } from "./main";
import Move, { MoveType } from "./core/move";

export default class Game {
    initialState: GameState;
    stateBefore: GameState;
    state: GameState;

    constructor() {
        this.initialState = {} as any;
        this.state = {} as any;
        this.stateBefore = {} as any;
    }

    load(state: GameState) {
        this.initialState = state;
        this.reset();
    }

    reset() {
        this.stateBefore = cloneState(this.initialState);
        this.state = cloneState(this.initialState);
    }

    playMove(moveIndex: number): Move[] {
        const moveTypes = [];
        this.stateBefore = cloneState(this.state);
        this.state.settings.live = true;
        const move = generateAllMoves(this.state)[moveIndex];
        moveTypes.push(move);
        if(move.moveType == MoveType.EndTurn) {
            // TODO how about updating the state in here? to compare start to end of current turn?
            this.endTurn();
            this.state.settings._playedMoves = [];
        }
        else {
            this.state.settings._playedMoves.push(move.moveType);
            const br = move.execute(this.state);
            // TODO figure out how to have the ai play reward moves (0/1)
            if(br.chainMoves?.length) {
                AIState.executeBestReward(this.state, br.chainMoves);
                moveTypes.push(br.chainMoves[0]);
            }
        }
        this.state.settings.live = false;
        return moveTypes;
    }
    
    /**
     * Generates random moves by selecting random moves and shuffling them
     */
    getGoodMoves(randomChance = 0.7): Move[] {
        const actual = cloneState(this.state);
        const doRandom = Math.random() < randomChance;

        const maxMoves = 7;
        const maxDepth = doRandom? 1 : 2;
        const maxArmyMoves = 10;
        const maxArmyDepth = doRandom? 1 : 2;

        // Use actual state to generate moves, or state will not match

        // const ecoMoves: BestMoves = null as any;
        const ecoMoves = evaluateBestMove(
            this.state, 
            (state: GameState) => generateEcoMoves(state)
                .sort(() => 0.5 - Math.random())
                // .slice(minMoves, minMoves + Math.floor(Math.random() * (maxMoves - minMoves + 1)))
                .slice(0, maxMoves)
                .filter(Boolean), 
                doRandom? () => Math.random() * 10 : evaluateEconomy, 
            maxDepth
        );

        if(!deepCompare(actual, this.state, 'state', true)) {
            throw Error('ECO COMPARE');
        }

        const undoChain: UndoCallback[] = [];

        if(ecoMoves?.moves.length) {
            logAndUndoMoves(ecoMoves.moves, this.state, false, undoChain);
        }

        const unitMoves = evaluateBestMove(
            this.state, 
            (state: GameState) => [
                ...getPovTribe(state)._units.map(x => UnitMoveGenerator.all(state, x)).flat(),
                ...UnitMoveGenerator.spawns(state),
            ].sort(() => 0.5 - Math.random()).slice(0, maxArmyMoves).filter(Boolean), 
            doRandom? () => Math.random() * 10 : evaluateArmy, 
            maxArmyDepth
        );

        undoChain.reverse().forEach(x => x());

        if(!deepCompare(actual, this.state, 'state', true)) {
            throw Error('ARMY COMPARE');
        }

        // if(ecoMoves) {
        //     const start = ecoMoves.moves.findIndex(x => x.id.startsWith('end'));
        //     ecoMoves.moves = ecoMoves.moves.slice(0, start);
        // }

        // if(unitMoves) {
        //     const start = unitMoves.moves.findIndex(x => x.id.startsWith('end'));
        //     unitMoves.moves = unitMoves.moves.slice(0, start);
        // }
        
        return [
            ...ecoMoves?.moves || [],
            ...unitMoves?.moves || [],
        ];
    }
    
    /**
     * Ends the current tribe's turn. Overwrites the passed state. Can NOT be undone 
     */
    endTurn() {
        // No more competing tribes remaining
        if(isGameOver(this.state) || isGameWon(this.state)) {
            this.state.settings._gameOver = true;
            return;
        }

        const newTribesMet: Set<TribeState> = new Set();
        
        const us = getPovTribe(this.state);
    
        // Move 0 is End Turn
        generateEndTurnMove(false).execute(this.state);

        // Reward discovering other tribes
        if(us._knownPlayers.length != this.state.settings.tribeCount) {
            // Check if we met any new tribes
            this.state._visibleTiles.forEach(x => {
                // If we can see any other tribe's unit, we have met them
                const standing = getUnitAtTile(this.state, x);
                if(standing 
                    && standing._owner != us.owner
                    && !us._knownPlayers.includes(standing._owner)
                    && !newTribesMet.has(this.state.tribes[standing._owner])
                ) {
                    newTribesMet.add(this.state.tribes[standing._owner]);
                }
            });
        
            // Reward stars for met tribes
            newTribesMet.forEach(them => {
                us._knownPlayers.push(them.owner);
                us._stars += getStarExchange(this.state, them);
    
                if(them._knownPlayers.includes(us.owner)) {
                    return;
                }
    
                // If they also too just met us
                for(const unit of us._units) {
                    if(this.state.tiles[unit._tileIndex].explorers.includes(them.owner)) {
                        them._knownPlayers.push(us.owner);
                        them._stars += getStarExchange(this.state, us);
                        break;
                    }
                }
            });
        }

        // TODO Add relations

        // Cycle turns without overflow
        const nextPov = () => {
            // Continue with next tribe
            this.state.settings._pov += 1;
            // If overflowing, go back to start
            if(this.state.settings._pov > this.state.settings.tribeCount) {
                this.state.settings._pov = STARTING_PLAYER;
            }
        }

        nextPov();

        let povTribe = getPovTribe(this.state);

        // Search for the next tribe
        while(povTribe._killedTurn > 0 || povTribe._resignedTurn > 0) {
            nextPov();
            povTribe = getPovTribe(this.state);
        }

        // If we are back at the start, a new turn has started
        if(this.state.settings._pov == STARTING_PLAYER) {
            this.state.settings._turn++;
        }
        
        this.updatePov();
    }

    /**
     * Updates the state's POV
     * @param state 
     * @param pov
     */
    public updatePov() {
        let pov = this.state.settings._pov;

        if(pov > this.state.settings.tribeCount) {
            pov = STARTING_PLAYER;
        }
       
        this.state.settings._pov = pov;
        this.state._visibleTiles = [];

        Object.values(this.state.tiles).forEach(tile => {
            if(tile.explorers.includes(pov)) {
                this.state._visibleTiles.push(tile.tileIndex);
            }
        });

        this.state._lighthouses = [
            0,
            this.state.settings.size - 1,
            this.state.settings.size * this.state.settings.size - 1,
            1 + this.state.settings.size * this.state.settings.size - this.state.settings.size
        ].filter(x => !this.state.tiles[x].explorers.includes(pov));

        this.state = {
            ...this.state,
            _potentialDiscovery: [],
            _potentialArmy: 0,
            _potentialTech: 0,
            _potentialEconomy: 0,
            _scoreArmy: 0,
            _scoreTech: 0,
            _scoreEconomy: 0,
            __: 0,
            ___: 0,
        }
        
        for(const strTileIndex in this.state.structures) {
            const structure = this.state.structures[strTileIndex];
            if(!structure || structure._owner > 0) continue;
            structure._potentialTerritory = getNeighborIndexes(this.state, structure.tileIndex, 1, true, true);
        }
    }
}