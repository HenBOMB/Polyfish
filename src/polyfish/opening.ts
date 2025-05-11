import { getPovTribe } from "../core/functions";
import Move from "../core/move";
import { MoveType } from "../core/types";
import { GameState } from "../core/states";
import { TribeType } from "../core/types";
import { MoveGenerator } from "../core/moves";
import EndTurn from "../core/moves/EndTurn";

export class Opening {
    /**
     * Returns a list of opening-best moves based on the current pov tribe and turn
     * @param state 
     * @param legalMoves legal moves to search instead
     * @returns returns [EndTurn] if NO opening moves were found
     */
    static recommend(state: GameState, legalMoves?: Move[]): Move[] {
        const pov = getPovTribe(state);
        const moveTypes = OpeningBook[pov.tribeType][state.settings._turn];
        const legal = legalMoves || MoveGenerator.legal(state);
        if(!moveTypes) return legal;
        const moves = legal.filter(x => moveTypes.some(y => x.moveType == y));
        return moves.length? moves : [new EndTurn()];
    }
}

const OpeningBook: { [key in TribeType]: { [key: number]: MoveType[] } } = {
    [TribeType.None]: { },
    [TribeType.Nature]: { },
    [TribeType.Imperius]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Summon, MoveType.Step],
    },
    [TribeType.AiMo]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Bardur]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Summon, MoveType.Step],
    },
    [TribeType.Aquarion]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Summon, MoveType.Step],
    },
    [TribeType.Elyrion]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Hoodrick]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Kickoo]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Luxidoor]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Oumaji]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Quetzali]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Vengir]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.XinXi]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Zebasi]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Yadakk]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Polaris]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    },
    [TribeType.Cymanti]: {
        0: [MoveType.Harvest, MoveType.Harvest, MoveType.Step],
        1: [MoveType.Step],
    }
}