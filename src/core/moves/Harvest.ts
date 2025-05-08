import { getPovTribe, isResourceVisible, isTechUnlocked } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType } from "../types";
import { GameState } from "../states";
import { ResourceSettings } from "../settings/ResourceSettings";
import { harvestResource } from "../actions";

export default class Harvest extends Move {
    constructor(target: number) {
        super(MoveType.Harvest, null, target, null);
    }

    execute(state: GameState): CallbackResult {
        return harvestResource(state, this.getTarget());
    }

    safeguard(state: GameState): 1 | null {
        const tileIndex = this.getTarget();
        const pov = getPovTribe(state);

        const resource = state.resources[tileIndex];
        
        if(!resource) {
            return null;// Logger.illegal(MoveType.Harvest, `Resource already harvested: ${tileIndex}`);
        }
    
        const settings = ResourceSettings[resource.id];
    
        // Cant afford it
        if((settings.cost || 0) > pov._stars) {
            return null;
        }
    
        // Resource requires a structure
        if(settings.structType) {
            return null;
        }
    
        // Cant harvest while there is a structure built
        if(state.structures[tileIndex]) {
            return null;
        }
    
        const tile = state.tiles[tileIndex];
    
        // If enemy is standing on resource, tile is blocked
        if(tile._unitOwner > 0 && tile._unitOwner != pov.tribeType) {
            return null;
        }
    
        // Resource is unharvestable
        if(!isTechUnlocked(pov, settings.techRequired)) {
            return null;
        }
    
        // Resource is limited by tech visibility
        if(!isResourceVisible(pov, resource.id)) {
            return null;
        }

        return 1;
    }
}