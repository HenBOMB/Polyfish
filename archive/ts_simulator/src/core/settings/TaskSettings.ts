import { getLighthouses, isTechUnlocked } from "../functions";
import { GameState } from "../states";
import { StructureType, TaskType, TechnologyType } from "../types";
import { TechnologyUnlockable } from "./TechnologySettings";

export const TaskSettings: Record<TaskType, { 
    techType?: TechnologyType;
    structureType: StructureType;
    task: (state: GameState) => boolean;
}> = {
    [TaskType.Pacifist]: {
        techType: TechnologyType.Meditation,
        structureType: StructureType.AltarOfPeace,
        task:(state: GameState) => {
            // TODO dont attack for 5 turns
            return false;
        }
    },
    [TaskType.Genius]: {
        techType: TechnologyType.Philosophy,
        structureType: StructureType.TowerOfWisdom,
        task: (state: GameState) => {
            if(!isTechUnlocked(state.tribes[state.settings.currentPlayerTurnId], TechnologyType.Philosophy)) {
                return false;
            }
            return state.tribes[state.settings.currentPlayerTurnId].tech_vanilla.length == Object.values(TechnologyUnlockable).length;
        }
    },
    [TaskType.Wealth]: {
        techType: TechnologyType.Trade,
        structureType: StructureType.EmperorsTomb,
        task: (state: GameState) => {
            if(!isTechUnlocked(state.tribes[state.settings.currentPlayerTurnId], TechnologyType.Trade)) {
                return false;
            }
            return state.tribes[state.settings.currentPlayerTurnId].cities.reduce((acc, cur) => acc + (cur.connectedToCapital ? 1 : 0), 0) > 4;
        }
    },
    [TaskType.Explorer]: {
        structureType: StructureType.EyeOfGod,
        task: (state: GameState) => {
            return getLighthouses(state).every(x => state.tiles[x].explorers.has(state.settings.currentPlayerTurnId));
        }
    },
    [TaskType.Killer]: {
        structureType: StructureType.GateOfPower,
        task: function (state: GameState): boolean {
            return state.tribes[state.settings.currentPlayerTurnId].kills > 9;
        }
    },
    [TaskType.Network]: {
        structureType: StructureType.GrandBazaar,
        task: function (state: GameState): boolean {
            return state.tribes[state.settings.currentPlayerTurnId].cities.filter(x => x.connectedToCapital).length > 4;
        }
    },
    [TaskType.Metropolis]: {
        structureType: StructureType.ParkOfFortune,
        task: function (state: GameState): boolean {
            return state.tribes[state.settings.currentPlayerTurnId].cities.some(x => x.level > 4);
        }
    }
}

export const IsStructureTask: Record<StructureType, boolean> = Object.values(TaskSettings)
    .reduce((a, b) => ({ ...a, [b.structureType]: true }), {}) as any;