import { UnitType } from "../core/types";


/**
 * Score based on the unit's tier i saw on youtube, may be redundant
 */
export const __SCORE_UNIT_TIER: Record<UnitType, number> = {
    [UnitType.None]:        0,
    // S tier
    [UnitType.Shaman]:      1.00,
    [UnitType.Rider]:       1.00,
    [UnitType.Hexapod]:     1.00,
    [UnitType.BattleSled]:  1.00,
    [UnitType.Amphibian]:   1.00,
    // A tier
    [UnitType.Archer]:      0.75,
    [UnitType.Knight]:      0.75,
    [UnitType.Dinghy]:      0.75,
    [UnitType.Cloak]:       0.75,
    [UnitType.Dagger]:      0.75,
    [UnitType.Pirate]:      0.75,
    [UnitType.Polytaur]:    0.75,
    [UnitType.Rammer]:      0.75,
    [UnitType.Scout]:       0.75,
    [UnitType.Centipede]:   0.75,
    [UnitType.Segment]:     0.75,
    [UnitType.FireDragon]:  0.75,
    [UnitType.BabyDragon]:  0.75,
    [UnitType.DragonEgg]:   0.75,
    [UnitType.Crab]:        0.75,
    // B tier
    [UnitType.Warrior]:     0.50,
    [UnitType.Defender]:    0.50,
    [UnitType.Catapult]:    0.50,
    [UnitType.Kiton]:       0.50,
    [UnitType.Tridention]:  0.50,
    [UnitType.Mooni]:       0.50,
    [UnitType.Raychi]:      0.50,
    [UnitType.Giant]:       0.50,
    [UnitType.Gaami]:       0.50,
    [UnitType.Phychi]:      0.50,
    [UnitType.Exida]:       0.50,
    [UnitType.Doomux]:      0.50,
    // C tier
    [UnitType.Swordsman]:   0.25,
    [UnitType.IceArcher]:   0.25,
    [UnitType.Bomber]:      0.25,
    [UnitType.Juggernaut]:  0.25,
    // F tier
    [UnitType.Raft]:        0.1,
    [UnitType.MindBender]:  0.01,
    [UnitType.IceFortress]: 0.01,
};
