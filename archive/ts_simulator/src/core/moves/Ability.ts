import Move from "../move";
import { MoveType } from "../types";

export default class Ability extends Move {
    constructor(src: number | null, target: number | null, type: number | null) {
        super(MoveType.Ability, src, target, type);
    }
}