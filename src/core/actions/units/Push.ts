import stepUnit from "./Step";
import { getTrueUnitAt, calaulatePushablePosition } from "../../functions";
import { CallbackResult, UndoCallback } from "../../move";
import { GameState } from "../../states";
import removeUnit from "./Remove";


export default function(state: GameState, tileIndex: number): CallbackResult {
    const pushed = getTrueUnitAt(state, tileIndex);

    if (!pushed) return null;

    const oldAttacked = pushed._attacked;
    const oldMoved = pushed._moved;
    const movedTo = calaulatePushablePosition(state, pushed);
    const rewards = [];

    let undoPush: UndoCallback = () => { };

    if (movedTo < 0) {
        undoPush = removeUnit(state, pushed);
    }
    else {
        const result = stepUnit(state, pushed, movedTo, true)!;
        rewards.push(...result.rewards);
    }

    return {
        rewards,
        undo: () => {
            undoPush();
            pushed._moved = oldMoved;
            pushed._attacked = oldAttacked;
        }
    };
}
