import { getPovTribe } from "../core/functions";
import Move from "../core/move";
import { MoveType } from "../core/types";
import { GameState } from "../core/states";
import { TribeType } from "../core/types";
import { MoveGenerator } from "../core/moves";

export class Opening {
    static recommend(state: GameState, legalMoves?: Move[]): number | null {
        const pov = getPovTribe(state);
        const moveTypes = OpeningBook[pov.tribeType][state.settings._turn];
        if(!moveTypes) return null;
        const index = (legalMoves || MoveGenerator.legal(state)).findIndex(x => moveTypes.includes(x.moveType));
        return index < 0? null : index;
    }
}

// The absolute best *suggested* starting moves a tribe can make on a turn
const OpeningBook: { [key in TribeType]: { [key: number]: MoveType[] } } = {
    [TribeType.None]: { },
    [TribeType.Nature]: { },
    [TribeType.Imperius]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.AiMo]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Aquarion]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Bardur]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Elyrion]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Hoodrick]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Kickoo]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Luxidoor]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Oumaji]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Quetzali]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Vengir]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.XinXi]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Zebasi]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Yadakk]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Polaris]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    },
    [TribeType.Cymanti]: {
        0: [MoveType.Harvest, MoveType.Harvest],
    }
}