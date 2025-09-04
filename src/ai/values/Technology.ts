import { TechnologyType, TribeType } from "../../core/types";
import Values from "./Values";

export default class TechnologyValues extends Values<TechnologyType> {

    /**
     * Scrore based on how much the technology is worth researching
     * @param tribeType 
     * @param values 
     */
    constructor(tribeType: TribeType) {
        super(tribeType, 'technology');
    }

    recommend() {
        // the -1 is because we cant really use them because they only replace the original tech
        // I just based these off really crude ideas, doesnt matter since they will be auto tuned by the AI
        this.load_values({
            // S
            [TechnologyType.Navigation]:    1.00,
            [TechnologyType.Mathematics]:   1.00,
            [TechnologyType.Trade]:         1.00,
            [TechnologyType.Roads]:         1.00,
            [TechnologyType.Strategy]:      1.00,
            [TechnologyType.Diplomacy]:     1.00,
            // A
            [TechnologyType.Riding]:        0.80,
            [TechnologyType.Chivalry]:      0.80,
            [TechnologyType.Organization]:  0.80,
            [TechnologyType.Fishing]:       0.80,
            [TechnologyType.Sailing]:       0.80,
            [TechnologyType.Ramming]:       0.80,
            // B
            [TechnologyType.Farming]:       0.60,
            [TechnologyType.Hunting]:       0.60,
            [TechnologyType.Archery]:       0.60,
            [TechnologyType.Forestry]:      0.60,
            [TechnologyType.Construction]:  0.60,
            [TechnologyType.Smithery]:      0.60,
            // C
            [TechnologyType.Aquatism]:      0.40,
            [TechnologyType.Climbing]:      0.40,
            [TechnologyType.Mining]:        0.40,
            // F
            [TechnologyType.FreeSpirit]:    0.20,
            [TechnologyType.Meditation]:    0.20,
            [TechnologyType.Philosophy]:    0.20,
            [TechnologyType.Spiritualism]:  0.20,
            
            // Unused
            [TechnologyType.None]:         -1.00,
            [TechnologyType.Unbuildable]:  -1.00,
            [TechnologyType.Oceantology]:  -1.00,
            [TechnologyType.PolarWarfare]: -1.00,
            [TechnologyType.ShockTactics]: -1.00,
            [TechnologyType.Spearing]:     -1.00,
            [TechnologyType.Amphibian]:    -1.00,
            [TechnologyType.FreeDiving]:   -1.00,
            [TechnologyType.Recycling]:    -1.00,
            [TechnologyType.Frostwork]:    -1.00,
            [TechnologyType.IceFishing]:   -1.00,
            [TechnologyType.Hydrology]:    -1.00,
            [TechnologyType.Pascetism]:    -1.00,
            [TechnologyType.Sledding]:     -1.00,
            [TechnologyType.Polarism]:     -1.00,
            [TechnologyType.ForestMagic]:  -1.00,
        })
    }
}
