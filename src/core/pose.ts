import { mkdirSync, writeFileSync, readdirSync } from "fs";
import { addPopulationToCity } from "./actions";
import { computeReachablePath, getCityAt, getCityOwningTile, getNeighborIndexes, getPovTribe } from "./functions";
import Move, { UndoCallback, CallbackResult } from "./move";
import { GameState, CityState, TribeState } from "./states";
import { StructureType, TerrainType } from "./types";

export default class PoseManager {
    constructor() {
        mkdirSync('data/poses');
    }

    load() {
        const files = readdirSync('data/poses');
        const poses = files.map(file => require(`../data/poses/${file}`));
        return poses;
    }

    save(state: GameState, moves: Move[]) {

    }
}