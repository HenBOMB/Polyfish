import stepUnit from "./Step";
import { getTrueUnitAt, calaulatePushablePosition } from "../../functions";
import { CallbackResult, UndoCallback } from "../../move";
import { GameState } from "../../states";
import removeUnit from "./Remove";


export default function pushUnit(state: GameState, tileIndex: number): CallbackResult {
    const pushed = getTrueUnitAt(state, tileIndex);

    if (!pushed) {
        return null;
    }

    const oldAttacked = pushed.attacked;
    const oldMoved = pushed.moved;
    const movedTo = calaulatePushablePosition(state, pushed);
    const rewards = [];

    let undoPush: UndoCallback = () => { };

    if (movedTo < 0) {
        undoPush = removeUnit(state, pushed);
    }
    else {
        if (getTrueUnitAt(state, movedTo)) {
            throw Error('tf');
        }
        const result = stepUnit(state, pushed, movedTo, true);
        rewards.push(...result.rewards);
        undoPush = result.undo;
    }

    return {
        rewards,
        undo: () => {
            undoPush();
            pushed.moved = oldMoved;
            pushed.attacked = oldAttacked;
        }
    };
}
