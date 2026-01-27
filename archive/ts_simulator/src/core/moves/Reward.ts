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

    execute(state: GameState) {
        const city = getCityAt(state, this.getSrc())!;
        // console.log(city);
        const rewardType = this.getType<RewardType>();
        const pov = getPovTribe(state);

        if (!city) {
            // console.log('xs', state.tiles[this.getSrc()]);
            // console.log('x', state.structures[this.getSrc()]);
            // console.log('x', state.tribes[state.settings.currentPlayerTurnId].cities);
            // throw Error('City not found');
        }

        let rewards = [];
        let undoReward = () => { };

        switch (rewardType) {
            case RewardType.Workshop:
                city.production++;
                undoReward = () => {
                    city.production--;
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
                city.population += 3;
                city.progress += 3;
                pov.score += 15; // 3 pop x 5 stars each
                undoReward = () => {
                    pov.score -= 15;
                    city.progress -= 3;
                    city.population -= 3;
                }
                break;
            case RewardType.BorderGrowth:
                city.borderSize++;
                const undoClaim = claimTerritory(state, getAdjacentIndexes(state, city.tileIndex, 2, undefined, true), city);
                rewards.push(...undoClaim.rewards);
                undoReward = () => {
                    undoClaim.undo();
                    city.borderSize--;
                }
                break;
            case RewardType.Park:
                city.production++;
                pov.score += 250;
                undoReward = () => {
                    pov.score -= 250;
                    city.production--;
                }
                break;
            case RewardType.SuperUnit:
                const resultSummon = summonUnit(
                    state,
                    TribeSettings[pov.type].uniqueSuperUnit || UnitType.Giant,
                    city.tileIndex
                )!;
                rewards.push(...resultSummon.rewards);
                undoReward = () => resultSummon.undo;
                break;
        }

        city.rewards.add(rewardType);

        return {
            rewards,
            undo: () => {
                city.rewards.delete(rewardType);
                undoReward();
            },
        };
    }
}
