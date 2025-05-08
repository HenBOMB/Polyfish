import { getCityAt, getNeighborTiles, getPovTribe, getTerritorry, isResourceVisible, isTechUnlocked } from "../functions";
import Move, { CallbackResult } from "../move";
import { MoveType, StructureType } from "../types";
import { GameState } from "../states";
import { buildStructure } from "../actions";
import { StructureSettings } from "../settings/StructureSettings";

export default class Structure extends Move {
    constructor(target: number, type: number) {
        super(MoveType.Build, null, target, type);
    }

    execute(state: GameState): CallbackResult {
        const strucType = this.getType<StructureType>();
        return buildStructure(state, strucType, this.getTarget());
    }

    safeguard(state: GameState): 1 | null {
        const pov = getPovTribe(state);
        const structType = this.getType<StructureType>();
        const settings = StructureSettings[structType];
        
        // ! Struct does not belong to this tribe
        if(settings.tribeType && settings.tribeType != pov.tribeType) {
            return null;
        }

        if(!settings.cost || (settings.cost || 0) > pov._stars) {
            return null;
        }

        if(!isTechUnlocked(pov, settings.techRequired)) {
            return null;
        }

        // ! Resource is required (and is visible)
        if(settings.resourceType) {
            if(state.resources[this.getTarget()]?.id != settings.resourceType) {
                return null;
            }
            if(!isResourceVisible(pov, settings.resourceType)) {
                return null;
            }
        }
        
        // EVAL: If doesnt require a resource but we are building ontop of a resource, assume bad move
        else if(state.resources[this.getTarget()]) {
            return null;
        }
        
        const tile = state.tiles[this.getTarget()];

        // ! If enemy is standing on resource, tile is blocked
        if(tile._unitOwner > 0 && tile._unitOwner != pov.owner) {
            return null;
        }
        
        // ! Task already built or not completed
        if(settings.task) {
            if(pov._builtUniqueStructures.some(x => x == structType) || !settings.task(pov, state.tiles)) {
                return null;
            }
            // EVAL Skip building it if it doesnt level up the city
            const city = pov._cities.find(x => tile._rulingCityIndex)!;
            const isLvlUp = (city._progress + (settings.rewardPop || 3)) >= city._level + 1;
            if(!isLvlUp) {
                return null;
            }
        }
        
        // ! Structure has already been built in this city's territory
        if(settings.limitedPerCity) {
            const city = getCityAt(state, tile._rulingCityIndex);
            const territory = getTerritorry(state, pov, city || undefined);
            if(territory.some(x => state.structures[x]?.id == structType && state.tiles[x]._rulingCityIndex == tile._rulingCityIndex)) {
                return null;
            }
        }
        
        // ! Adjacent tiles do not contain matching structure
        if(settings.adjacentTypes != undefined &&
            !getNeighborTiles(state, tile.tileIndex).some(x => state.structures[x.tileIndex]? settings.adjacentTypes!.includes(state.structures[x.tileIndex]!.id) : false)
        ) {
            return null;
        }

        return 1;
    }
}