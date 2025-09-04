import { getPovTribe } from "../core/functions";
import Move from "../core/move";
import { MoveType } from "../core/types";
import { GameState } from "../core/states";
import { TribeType } from "../core/types";

export class Opening {
    /**
     * Returns a list of opening-best moves based on the current pov tribe and turn
     * @param state 
     * @param legal legal moves to search in
     * @returns returns a list of move indexes that are book moves
     */
    static recommend(state: GameState, legal: Move[]): Move[] {
        const pov = getPovTribe(state);
        const moveTypes = OpeningBook[pov.tribeType][state.settings._turn];
        if(!moveTypes || legal.length < 2) return [];
        return legal.reduce((a: any, b, i) => ([
            ...a, ...(moveTypes.some(y => b.moveType == y)? [i] : [])
        ]), []).map((x: number) => legal[x]);
    }
}

/**
 * Opening book, each record contains the turn at which to apply a good recommended move. 
 * May not always be accurate, since the terrain generation is random.
 * TODO: make dynamic like `ai.values`
 */
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