import { TribeType, UnitType } from "../../core/types";
import Values from "./Values";

export default class UnitValues extends Values<UnitType> {

    /**
     * Score based on how strong the unit generally is on the battlefield
     * @param tribeType 
     * @param values 
     */
    constructor(tribeType: TribeType) {
        super(tribeType, 'units');
    }

    recommend() {
        this.load_values({
            [UnitType.None]:       -1.00,

            // Super Units //

            // S tier
            [UnitType.Shaman]:      1.00,
            // A tier
            [UnitType.FireDragon]:  0.95,
            [UnitType.BabyDragon]:  0.90,
            [UnitType.DragonEgg]:   0.85,
            [UnitType.Centipede]:   0.83,
            [UnitType.Segment]:     0.81,
            [UnitType.Crab]:        0.80,
            // B tier
            [UnitType.Giant]:       0.74,
            [UnitType.Gaami]:       0.70,
            // C tier
            [UnitType.Juggernaut]:  0.50,

            // Spawnable Units //
            
            // S tier
            [UnitType.Rider]:       0.60,
            [UnitType.Hexapod]:     0.60,
            [UnitType.BattleSled]:  0.50,
            [UnitType.Amphibian]:   0.60,
            // A tier
            [UnitType.Archer]:      0.47,
            [UnitType.Knight]:      0.46,
            [UnitType.Cloak]:       0.45,
            [UnitType.Dinghy]:      0.43,
            [UnitType.Dagger]:      0.43,
            [UnitType.Pirate]:      0.43,
            [UnitType.Polytaur]:    0.42,
            [UnitType.Rammer]:      0.41,
            [UnitType.Scout]:       0.40,
            // B tier
            [UnitType.Warrior]:     0.39,
            [UnitType.Defender]:    0.38,
            [UnitType.Catapult]:    0.37,
            [UnitType.Kiton]:       0.36,
            [UnitType.Tridention]:  0.36,
            [UnitType.Mooni]:       0.36,
            [UnitType.Raychi]:      0.35,
            [UnitType.Phychi]:      0.35,
            [UnitType.Exida]:       0.35,
            [UnitType.Doomux]:      0.34,
            // C tier
            [UnitType.Swordsman]:   0.29,
            [UnitType.IceArcher]:   0.29,
            [UnitType.Bomber]:      0.27,
            // F tier
            [UnitType.MindBender]:  0.15,
            [UnitType.IceFortress]: 0.16,
            [UnitType.Raft]:        0.16,
        })
    }
}