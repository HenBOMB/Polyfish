import { xorCity } from "../../zobrist/hasher";
import { getPovTribe } from "../functions";
import { Branch, CallbackResult } from "../move";
import { EconMovesGenerator } from "../moves";
import { GameState, CityState } from "../states";

export default function(state: GameState, city: CityState, amount: number): Branch {
    const pov = getPovTribe(state);

    const cityStruct = state.structures[city.tileIndex]!;

    city.population += amount;
    city.progress += amount;

    const next = city.level + 1;

    if (city.progress >= next) {
        const lvl = city.level;

        cityStruct.level++;
        city.level++;
        city.progress -= next;
        city.production++;

        let rewards = EconMovesGenerator.rewards(city);
        let lol = false;
        let amountScore = (city.level > 1 ? 50 - (city.level - 2) * 5 : 0) + amount * 5;

        if (city.progress - next >= (next + 1)) {
            // useful for debugging multithread
            // if it prints it means there is something wrong..
            // it should ONLY print when connecting cities and getting a huge pop increase
            console.warn('MEGA CHAIN!');
            // process.exit(1)
            lol = true;
            cityStruct.level++;
            city.level++;
            city.progress -= next + 1;
            city.production++;
            rewards.push(...EconMovesGenerator.rewards(city));
            amountScore += (city.level > 1 ? 50 - (city.level - 2) * 5 : 0) + amount * 5;
        }

        xorCity.level(state, city, lvl, city.level);

        pov.score += amountScore;

        return {
            rewards,
            undo: () => {
                pov.score -= amountScore;

                xorCity.level(state, city, city.level, lvl);

                if (lol) {
                    city.production--;
                    city.progress += next + 1;
                    city.level--;
                    cityStruct.level--;
                }

                city.production--;
                city.progress += next;
                city.level--;
                cityStruct.level--;

                city.progress -= amount;
                city.population -= amount;
            },
        };
    }

    return {
        rewards: [],
        undo: () => {
            city.progress -= amount;
            city.population -= amount;
        }
    };
}
