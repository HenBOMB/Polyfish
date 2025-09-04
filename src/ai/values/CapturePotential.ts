import { TribeType, UnitType } from "../../core/types";
import Values from "./Values";

export default class CapturePotentialValues extends Values<UnitType> {

    /**
     * Score based on how effective a unit it is closer enemy cities
     * @param tribeType
     * @param values
     */
    constructor(tribeType: TribeType) {
        super(tribeType, 'capture-potential');
    }

    recommend() {
        this.load_values({
            [UnitType.None]:        0.00,
            // Get them on their capital to dominate
            [UnitType.Crab]:        0.95,
            [UnitType.Juggernaut]:  0.94,
            [UnitType.Giant]:       0.90,
            [UnitType.Gaami]:       0.40,
            [UnitType.Shaman]:      0.40,
            [UnitType.Dinghy]:      0.26,
            [UnitType.Cloak]:       0.25,
            [UnitType.Dagger]:      0.17,
            [UnitType.Pirate]:      0.14,
            [UnitType.Hexapod]:     0.25,
            [UnitType.Amphibian]:   0.22,
            [UnitType.Rider]:       0.20,
            [UnitType.Tridention]:  0.20,
            [UnitType.Swordsman]:   0.17,
            [UnitType.BattleSled]:  0.16,
            [UnitType.Knight]:      0.14,
            [UnitType.Polytaur]:    0.13,
            [UnitType.Warrior]:     0.12,
            [UnitType.Defender]:    0.11,
            [UnitType.Rammer]:      0.11,
            [UnitType.Scout]:       0.11,
            [UnitType.Kiton]:       0.11,
            [UnitType.Raychi]:      0.15,
            [UnitType.DragonEgg]:   0.10,
            [UnitType.BabyDragon]:  0.10,
            [UnitType.FireDragon]:  0.10,
            [UnitType.Doomux]:      0.10,
            [UnitType.Phychi]:      0.10,
            [UnitType.Exida]:       0.10,
            [UnitType.Centipede]:   0.10,
            [UnitType.Segment]:     0.10,
            // Generally avoid because they are weak
            [UnitType.Archer]:      0.07,
            [UnitType.Bomber]:      0.06,
            [UnitType.Catapult]:    0.01,
            [UnitType.MindBender]:  0.05,
            [UnitType.Mooni]:       0.06,
            [UnitType.IceFortress]: 0.09,
            [UnitType.IceArcher]:   0.06,
            [UnitType.Raft]:        0.06,
        })
    }
}