import { GameState, TribeState } from "./core/states";
import AIState from "./aistate";
import { STARTING_OWNER_ID } from "./core/gameloader";
import UnitMoveGenerator, { generateAllMoves, generateEcoMoves, generateEndTurnMove, UndoCallback } from "./core/moves";
import { cloneState, getCityProduction, getNeighborIndexes, getPovTribe, getStarExchange, getUnitAtTile, isFrozen, isGameOver, isGameWon, tryDiscoverRewardOtherTribes } from "./core/functions";
import { evaluateArmy, evaluateEconomy } from "./eval/eval";
import { evaluateBestMove, logAndUndoMoves } from "./eval/evalMoves";
import { deepCompare } from "./main";
import Move, { MoveType } from "./core/move";
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
        // TODO Add relations? (for polytopia default bots)

        // Cycle turns without overflow
        const nextPov = () => {
            // Continue with next tribe
            this.state.settings._pov += 1;
            // If overflowing, go back to start
            if(this.state.settings._pov > this.state.settings.tribeCount) {
                this.state.settings._pov = STARTING_OWNER_ID;
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
        if(this.state.settings._pov == STARTING_OWNER_ID) {
            this.state.settings._turn++;
        }

        // Properly wait for new turn and check if the game has ended
        if(isGameOver(this.state)) {
            this.state.settings._gameOver = true;
            return;
        }

        // TODO BOOST LOGIC
        // The Shaman, a unit unique to the Cymanti tribe, can boost friendly units. Boosted units get +0.5 attack and +1 movement. This effect lasts until the boosted unit attacks another unit, is attacked, uses most abilities, examines a ruin, captures a village/city, or is poisoned.
        // TODO POISON LOGIC
        // Poison is applied by Cymanti units and buildings. It reduces defence by 30%, prevents the unit from being healed, prevents the unit from receiving any defence bonus, and causes the unit to drop spores (on land) or Algae (in water) upon death. Poison can be removed by healing once. This can be through self-healing, a Mind Bender, or a Mycelium. When healed in this way, the poison is removed but the unit does not get any health back.

        // Reward tribe with its production
        povTribe._stars += getCityProduction(this.state, ...povTribe._cities);
        
        // Update all unit states
		for (let i = 0; i < povTribe._units.length; i++) {
			const unit = povTribe._units[i];

            // Frozen units get unfrozen but that consumes their turn
            // NOTE not in wiki but i assume this is how it works
			if(isFrozen(unit)) {
                // Remove ONLY the frozen effect
				unit._effects.splice(unit._effects.indexOf(EffectType.Frozen), 1);
				unit._moved = unit._attacked = true;
				continue;
			}

			unit._moved = unit._attacked = false;
		}

        // Trigger disovery if some other tribes moved into our visible terrain
        tryDiscoverRewardOtherTribes(this.state);

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
            pov = STARTING_OWNER_ID;
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