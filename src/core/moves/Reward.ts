import { getCityAt, getAdjacentIndexes, getPovTribe } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, RewardType } from "../types";
import { GameState } from "../states";
import { UnitType } from "../types";
import { TribeSettings } from "../settings/TribeSettings";
import { predictExplorer } from "../../ai/prediction";
import { gainStars, spendStars } from "../actions";
import claimTerritory from "../actions/ClaimTerritory";
import { discoverTiles } from "../actions/DiscoverTiles";
import summonUnit from "../actions/units/Summon";

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
            case RewardType.PopGrowth:
                city._population += 3;
                city._progress += 3;
                pov._score += 15; // 3 pop x 5 stars each
                undoReward = () => {
                    pov._score -= 15;
                    city._progress -= 3;
                    city._population -= 3;
                }
                break;
            case RewardType.BorderGrowth:
                city._borderSize++;
                const undoClaim = claimTerritory(state, getAdjacentIndexes(state, city.tileIndex, 2, undefined, true));
                rewards.push(...undoClaim.rewards);
                undoReward = () => {
                    undoClaim.undo();
                    city._borderSize--;
                }
            break;
            case RewardType.Park:
                city._production++;
                pov._score += 250;
                undoReward = () => {
                    pov._score -= 250;
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
