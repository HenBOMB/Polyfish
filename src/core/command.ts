import Move, { MoveType } from "./move";
import Step from "./moves/Step";
import Summon from "./moves/Summon";

// TODO
export function Command(moveType: MoveType, src: number, target: number, type: number): Move {
    switch (moveType) {
        case MoveType.Step:
            return new Step(src, target, type);

        case MoveType.Summon:
            // Upgrade?
            return new Summon(src, target, type);
    
        default:
            throw new Error(`Invalid move: ${moveType} (${MoveType[moveType]})`);
    }
}