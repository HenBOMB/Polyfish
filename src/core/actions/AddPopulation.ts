import { xorCity } from "../../zobrist/hasher";
import { getPovTribe } from "../functions";
import { Branch, CallbackResult } from "../move";
import { EconMovesGenerator } from "../moves";
import { GameState, CityState } from "../states";


export default function(state: GameState, city: CityState, amount: number): Branch {
    const pov = getPovTribe(state);

    const cityStruct = state.structures[city.tileIndex]!;

    city._population += amount;
    city._progress += amount;

    const next = city._level + 1;

    if (city._progress >= next) {
        const lvl = city._level;

        cityStruct._level++;
        city._level++;
        city._progress -= next;
        city._production++;

        let rewards = EconMovesGenerator.rewards(city);
        let lol = false;
        let amountScore = (city._level > 1 ? 50 - (city._level - 2) * 5 : 0) + amount * 5;

        if (city._progress - next >= (next + 1)) {
            console.warn('MEGA CHAIN!');
            lol = true;
            cityStruct._level++;
            city._level++;
            city._progress -= next + 1;
            city._production++;
            rewards.push(...EconMovesGenerator.rewards(city));
            amountScore += (city._level > 1 ? 50 - (city._level - 2) * 5 : 0) + amount * 5;
        }

        xorCity.level(state, city, lvl, city._level);

        pov._score += amountScore;

        return {
            rewards,
            undo: () => {
                pov._score -= amountScore;

                xorCity.level(state, city, city._level, lvl);

                if (lol) {
                    city._production--;
                    city._progress += next + 1;
                    city._level--;
                    cityStruct._level--;
                }

                city._production--;
                city._progress += next;
                city._level--;
                cityStruct._level--;

                city._progress -= amount;
                city._population -= amount;
            },
        };
    }

    return {
        rewards: [],
        undo: () => {
            city._progress -= amount;
            city._population -= amount;
        }
    };
}
