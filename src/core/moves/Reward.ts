import { getCityAt, getNeighborIndexes, getPovTribe } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, RewardType } from "../types";
import { GameState } from "../states";
import { UnitType } from "../types";
import { TribeSettings } from "../settings/TribeSettings";
import { predictExplorer } from "../../eval/prediction";
import { discoverTiles, gainStars, spendStars, summonUnit } from "../actions";

export default class Reward extends Move {
    constructor(src: number, type: number) {
        super(MoveType.Reward, src, null, type);
    }

    execute(state: GameState): CallbackResult {
        const city = getCityAt(state, this.getSrc())!;
        const rewardType = this.getType<RewardType>();
        const pov = getPovTribe(state);

        let rewards = [];
        let undoReward = () => { };

        switch (rewardType) {
            case RewardType.Workshop:
                city._production++;
                undoReward = () => {
                    city._production--;
                }
                break;
            case RewardType.Explorer:
                const resultDiscover = discoverTiles(state, null, predictExplorer(state, city.tileIndex))!;
                undoReward = resultDiscover.undo;
                rewards.push(...resultDiscover.rewards);
                break;
            case RewardType.CityWall:
                city._walls = true;
                undoReward = () => {
                    city._walls = false;
                };
                break;
            case RewardType.Resources:
                undoReward = gainStars(state, 5);
                break;
            case RewardType.PopulationGrowth:
                city._population += 3;
                city._progress += 3;
                undoReward = () => {
                    city._progress -= 3;
                    city._population -= 3;
                }
                break;
            case RewardType.BorderGrowth:
                city._borderSize++;
                const undoDiscover = discoverTiles(state, null, getNeighborIndexes(state, city.tileIndex, 2))!;
                rewards.push(...undoDiscover.rewards);
                undoReward = () => {
                    undoDiscover.undo();
                    city._borderSize--;
                }
            break;
            case RewardType.Park:
                city._production++;
                pov._score += 300;
                undoReward = () => {
                    pov._score -= 300;
                    city._production--;
                }
                break;
            case RewardType.SuperUnit:
                const resultSummon = summonUnit(
                    state, 
                    TribeSettings[pov.tribeType].uniqueSuperUnit || UnitType.Giant, 
                    city.tileIndex
                )!;
                rewards.push(...resultSummon.rewards);
                undoReward = () => resultSummon.undo;
                break;
        }
        
        city._rewards.add(rewardType);
        
        return {
            rewards,
            undo: () => {
                city._rewards.delete(rewardType);
                undoReward();
            },
        };
    }
}
