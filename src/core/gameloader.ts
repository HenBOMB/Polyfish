import { exec } from "child_process";
import {
    ClimateType,
    ModeType,
    ResourceType,
    RewardType,
    StructureType,
    TechnologyType,
    TerrainType,
    TribeType,
    UnitType,
} from "./types";
import { UnitState, CityState, GameState, TribeState, TileState, ResourceState, StructureState, DiplomacyRelationState, GameSettings, DefaultGameSettings, PartialGameSettings, Coords } from "./states";
import { isResourceVisible, getAdjacentTiles, getAdjacentIndexes, isWaterTerrain, isIceTerrain, calculateInitialTribeScore, getHomeCity, getPovTribe, getUnitAt } from "./functions";
import { readFileSync, writeFileSync } from "fs";
import { UnitSettings } from "./settings/UnitSettings";
import { predictBestNextCityReward, predictOuterFogTerrain, predictVillages } from "../ai/prediction";
import { TribeSettings } from "./settings/TribeSettings";
import summonUnit from "./actions/units/Summon";
import { xorState } from "../zobrist/hasher";
import { ResourceSettings } from "./settings/ResourceSettings";
import { StructureSettings } from "./settings/StructureSettings";

export const STARTING_OWNER_ID = 1;
export const MAX_SEED = 10;
// Standard max turns when loaded live games
export const MAX_TURNS = 50;

function parseSettings(settings?: PartialGameSettings | GameSettings): GameSettings {
    settings = {
        ...DefaultGameSettings,
        ...(settings || {})
    };
    settings.size = Number(settings.size);
    return settings as GameSettings;
}

export default class GameLoader {
    public currentState: GameState;

    constructor() {
        this.currentState = { } as any;
    }

    private defaultState(): GameState {
        return {
            settings: {
                size: 0,
                tileCount: 0,
                turn: 0,
                maxTurns: 0,
                currentPlayerTurnId: 0,
                mode: ModeType.Perfection,
                _lastPlayerTurnId: -1,
                _areYouSure: false,
                _maxTribeCount: 0,
                _gameOver: false,
                _recentMoves: [],
                _pendingRewards: [],
                _fow: true,
            },
            tiles: {},
            structures: {},
            resources: {},
            tribes: {},
            _visibleTiles: {}
        }
    }

    private async readLiveGameData(): Promise<any> {
        return new Promise((resolve, reject) => {
            exec("bash scan.sh -y", (error: any, stdout: string, stderr: any) => {
                if (error) {
                    if (error.code == 1) {
                        return reject(stdout.trim());
                    }
                    if (!stdout.length && (error || stderr)) {
                        return reject(error || stderr);
                    }
                }
                try {
                    console.log('saved');
                    writeFileSync("data/gamestate.json", JSON.stringify(JSON.parse(stdout), null, 4));
                    resolve(JSON.parse(stdout));
                } catch (error) {
                    reject(error);
                }
            });
        });
    }

    private loadGame(state: GameState) {
        this.currentState = state;
        // TODO: predictions disabled
        // this.updatePredictions(state);

        // If FOW is disabled, then tribes shouldn't claim discovering other tribes
        if (!state.settings._fow) {
            const tribesObj = Object.values(state.tribes);
            for(const tribe of tribesObj) {
                tribesObj.forEach(x => {
                    if (x.id != tribe.id) {
                        tribe.knownPlayers.add(x.id);
                    }
                });
            }
            // if undefined
            state.settings._fow = false;
        }

        const pov = state.settings.currentPlayerTurnId;

        for(const tribe of Object.values(state.tribes)) {
            state.settings.currentPlayerTurnId = tribe.id
            tribe._hash = xorState(state);
        }

        state.settings.currentPlayerTurnId = pov;
    }

    public async loadLive(settings?: PartialGameSettings): Promise<GameState> {
        let state = await this.readLiveGameData().catch((err) => {
            console.log(err);
            return null;
        }) as GameState;

        if (!state) {
            if (!settings?.fallback) {
                throw new Error("NO LIVE DATA FOUND");
            }
            else {
                state = JSON.parse(readFileSync(settings.fallback, 'utf-8')) as GameState;
            }
        }

        const parseCoords = (coords: any) => Coords.from(coords[0], coords[1], state);

        // TODO All raw coords are parsed into <Coords>

        state.tribes = Object.values(state.tribes)
            .reduce((acc: Record<number, TribeState>, tribeState: TribeState) => {
                return { 
                    ...acc, [tribeState.id]: {
                        ...tribeState,
                        knownPlayers: new Set(tribeState.knownPlayers),
                        builtUniqueImprovements: new Set(tribeState.builtUniqueImprovements),
                        startingTileCoords: parseCoords(tribeState.startingTileCoords),
                        _hash: 0n,
                    } as TribeState 
                }
            }
        , {});

        state._visibleTiles = { };

        (state as any)._hiddenResources = { };

        if (state.settings.turn > MAX_TURNS) {
            console.log(`WARN: turn is "${state.settings.turn}", set to 0`);
            state.settings.turn = 0;
        }

        // ! used only for resources custom visibility
        const playerTribe = getPovTribe(state);

        for (const _ in state.tiles) {
            const tileState = state.tiles[_];

            tileState.coords = parseCoords(tileState.coords);
            
            // Tile
            if (tileState.rulingCityCoords) {
                tileState.rulingCityCoords = parseCoords(tileState.rulingCityCoords);
            }

            tileState.explorers = new Set(tileState.explorers);
        }

        for (const _ in state.tiles) {
            const tileState = state.tiles[_];
            
            // Resource
            const resource = state.resources[tileState.coords.idx];
            if (resource) {
                if (!ResourceSettings[resource.type]) {
                    console.log(tileState);
                    console.log(TerrainType[tileState.type]);
                    console.log(ClimateType[tileState.climate]);
                    throw Error(`ResourceType type=${resource.type} does isn't registered!`);
                }

                if (!isResourceVisible(playerTribe, resource.type)) {
                    (state as any)._hiddenResources[resource.tileIndex] = resource.type;
                }
            }

            // Structure
            const structure = state.structures[tileState.coords.idx];
            if (structure) {
                if (!StructureSettings[structure.type]) {
                    throw Error(`StructureType type=${structure.type} does isn't registered!`);
                }
            }
        }

        for(const tribeId in state.tribes) {
            const tribe = state.tribes[tribeId];

            // Unit
            for (let i = 0; i < tribe.units.length; i++) {
                const unit = tribe.units[i];
                
                unit.effects = new Set(unit.effects);
                unit.coords = parseCoords(unit.coords);
                unit.homeCoords =  unit.homeCoords && unit.homeCoords?.x != -1? parseCoords(unit.homeCoords) : undefined;
                unit.prevCoords = parseCoords(unit.prevCoords);
                
                if (!UnitSettings[unit.type]) {
                    throw Error(`UnitType type=${unit.type} does isn't registered!`);
                }
            }

            // City
            for(const city of tribe.cities) {
                city.rewards = new Set(city.rewards);
                city._territory = getAdjacentTiles(state, city.tileIndex, city.borderSize)
                    .filter(x => x.rulingCityCoords?.idx == city.tileIndex).map(x => x.coords.idx);
            }
        }

        this.loadGame(state);

        return state;
    }

    public async randomNotation(settings?: GameSettings, seed?: number): Promise<string> {
        if (seed === undefined) {
            seed = Math.floor(Math.random() * MAX_SEED);
        }

        if (!settings) {
            settings = parseSettings(settings);
        }

        const mapdata: { type: string, tribe: string, above: string | null, road: boolean }[] = JSON.parse(await new Promise((resolve, reject) => {
            const cmd = `.venv/bin/python mapgen/main.py --seed ${seed} --size ${settings.size} --tribes ${settings.tribes.map(x => TribeType[x]).join(" ")}`
            exec(cmd, (error: any, stdout: string, stderr: any) => {
                if (error) {
                    // console.log(error);
                    return reject(error || stderr);
                }
                resolve(stdout.trim());
            });
        }));

        // Convert mapdata to notation for simplicity
        return [
            // Settings
            [`${ModeType[settings.mode!].toLowerCase()},0,${settings.maxTurns},1`],
            // Tribes
            settings.tribes.map(x => TribeType[x].slice(0, 2).toLowerCase()),
            // Climate
            mapdata.map(x => x.tribe.slice(0, 2).toLowerCase()),
            // Terrain Type
            mapdata.map(x => {
                switch (x.type) {
                    case 'village':
                    case 'ruin':
                    case 'ground':
                        return '-';
                    default:
                        return x.type[0];
                }
            }),
            // Resource y/n
            mapdata.map(x => x.above ? 'y' : '-'),
            // Villages & Capitals & Ruins TODO CAPITALS
            mapdata.map((x) =>
                x.above == 'capital'? x.tribe.slice(0, 2).toLowerCase() :
                x.above == 'ruin' ? 'rs' :
                x.above == 'starfish' ? 'sf' :
                x.above == 'village' ? 'vv' :
                '--'
            ),
        ].map(x => x.join('')).join(';');
    }

    public async loadRandom(psettings?: PartialGameSettings, verbose = true) {
        const settings = parseSettings(psettings);
        // Safeguard for inconsistency map generation
        let tries = 1000
        let seed = psettings?.seed? psettings?.seed : Math.floor(Math.random() * MAX_SEED);

        while(tries > 0) {
            try {
                const not = await this.randomNotation(settings, seed).catch(() => null);
                if (!not) throw 'err';
                this.loadNotation(not, settings);
                if (verbose) {
                    console.log('SEED', seed);
                }
                return [this.loadNotation(not, settings)];
            } catch (error) {
                console.log(error);
                tries--;
                seed++;
            }
        }

        console.log(`TRIED ${1000} TIMES AND ALL FAILED! USING EMERGENCY STATE!`);
    }

    public loadSave(filename: string) {
        const state = JSON.parse(readFileSync(`data/${filename}.json`, 'utf-8')) as GameState;
        this.loadGame(state);
    }

    public loadNotation(notation: string, someSettings?: GameSettings) {
        const [settingsRaw, tribeRaw, climateRaw, terrainRaw, resourceRaw, structuresRaw] = notation.split(';');

        // Settings (mode, turn, maxturn, pov)
        const notSettings = settingsRaw.split(',');

        const pov = Number(notSettings[3]);

        const TribeMap: { [key: string]: TribeType } = {
            ai: TribeType.AiMo,
            aq: TribeType.Aquarion,
            ba: TribeType.Bardur,
            el: TribeType.Elyrion,
            ho: TribeType.Hoodrick,
            im: TribeType.Imperius,
            ki: TribeType.Kickoo,
            lu: TribeType.Luxidoor,
            ou: TribeType.Oumaji,
            qu: TribeType.Quetzali,
            ve: TribeType.Vengir,
            xi: TribeType.XinXi,
            ze: TribeType.Zebasi,
            ya: TribeType.Yadakk,
            po: TribeType.Polaris,
            cy: TribeType.Cymanti,
        };

        const TerrainMap: { [key: string]: TerrainType } = {
            '-': TerrainType.Field,
            'p': TerrainType.Field, // plains
            'l': TerrainType.Field, // land
            m: TerrainType.Mountain,
            i: TerrainType.Ice,
            f: TerrainType.Forest,
            w: TerrainType.Water,
            o: TerrainType.Ocean,
        };

        // Set tribes

        const tribes = (tribeRaw.match(/.{1,2}/g)!.map(x => TribeMap[x]) as unknown as TribeType[]).reduce((arr, type, i) => {
            const owner = i + 1;
            return {
                ...arr,
                [owner]: {
                    _hash: 0n,
                    id: owner,
                    username: owner == pov? "Player" : TribeType[type],
                    bot: owner != pov,
                    type,
                    score: 0,
                    stars: 5,
                    killedTurn: -1,
                    resignedTurn: -1,
                    killerId: -1,
                    tech_vanilla: [TechnologyType.Unrequired, ...TribeSettings[type].startingTech? [TribeSettings[type].startingTech] : []].map(x => ({
                        type: x as TechnologyType,
                        discovered: true,
                    })),
                    kills: 0,
                    casualties: 0,
                    tasks: [],
                    builtUniqueImprovements: new Set(),
                    cities: [],
                    units: [],
                    resources: [],
                    structures: [],
                    knownPlayers: new Set(),
                    relations: [],
                    // TODO never assigned
                    startingTileCoords: new Coords(-1, null),
                } as TribeState,
            };
        }, {}) as { [key: number]: TribeState };
        
        someSettings = parseSettings(someSettings);

        const state: GameState = {
            ...this.defaultState(),
            settings: {
                mode: someSettings.mode!,
                size: Math.sqrt(Number(climateRaw.length / 2)),
                tileCount: climateRaw.length,
                turn: Number(notSettings[1]),
                _lastPlayerTurnId: -1,
                maxTurns: someSettings.maxTurns!,
                currentPlayerTurnId: pov,
                _areYouSure: false,
                _maxTribeCount: Object.keys(tribes).length,
                _gameOver: false,
                _recentMoves: [],
                _pendingRewards: [],
                _fow: someSettings.fow
            },
            tribes
        };

        (state as any)._hiddenResources = { };

        for(const owner in state.tribes) {
            state.tribes[owner].relations = Object.values(state.tribes).reduce((acc: any, tribe) => ({
                ...acc,
                [tribe.id]: {
                    state: 0,
                    lastAttackTurn: -1,
                    embassyLevel: -1,
                    lastPeaceBrokenTurn: -1,
                    firstMeet: -1,
                    embassyBuildTurn: -1,
                    previousAttackTurn: -1,
                }
            }), {});
        }

        // TODO: always assuming this is turn 0

        const lighthouses = [
            0,
            state.settings.size - 1,
            state.settings.size * state.settings.size - 1,
            1 + state.settings.size * state.settings.size - state.settings.size
        ];

        // Set tiles

        const climateTypes = climateRaw.match(/.{1,2}/g)!;
        const terrainTypes = terrainRaw.match(/./g)!;
        const explorerOwners = state.settings._fow? [] : Object.values(tribes).map(x => x.id);

        for(let i = 0; i < climateTypes.length; i++) {
            const climate = TribeMap[climateTypes[i]]? ClimateType[TribeType[TribeMap[climateTypes[i]]] as any] as unknown as ClimateType : ClimateType.Nature;

            state.tiles[i] = {
                owner: 0,
                climate,
                // If tile is ocean tile, then its nature??
                type: TerrainMap[terrainTypes[i]],
                explorers: new Set(explorerOwners),
                hasRoad: false,
                hasRoute: false,
                hadRoute: false,
                capitalOf: -1,
                skinType: -1,
                coords: new Coords(i, state)
            }
        }

        // Set spawning cities, ruins and starfish

        for(let i = 0; i < structuresRaw.length; i += 2) {
            const tileIndex = i / 2;
            const structureOrTribeType = structuresRaw.substring(i, i + 2);

            if (structureOrTribeType == 'vv') {
                state.structures[tileIndex] = {
                    type: StructureType.Village,
                    level: 1,
                    founded: 0,
                    score: 0,
                    tileIndex,
                }
            }
            else if (TribeMap[structureOrTribeType]) {
                const tribeType = TribeMap[structureOrTribeType];
                const territory = [tileIndex, ...getAdjacentIndexes(state, tileIndex, 1, false, true)];
                // const tribe = Object.values(state.tribes).find(x => x.tribeType == tribeType)!;
                const tribe = Object.values(state.tribes)
                    .filter(x => x.type === tribeType)
                    .find(x => x.cities.length === 0)!;

                for(const tile of territory) {
                    state.tiles[tile] = {
                        ...state.tiles[tile],
                        owner: tribe.id,
                        capitalOf: tribe.id,
                        rulingCityCoords: new Coords(tileIndex, state),
                    }
                }

                // Reveal surrounding land
                if (state.settings._fow) {
                    for(const tile of [
                        tileIndex,
                        ...getAdjacentIndexes(state, tileIndex, 2, false, true).filter(x => !lighthouses.includes(x))
                    ]) {
                        state.tiles[tile].explorers.add(tribe.id);
                    }
                }

                const cityData: CityState = {
                    tileIndex,
                    name: `${TribeType[tribeType]} ${state.tiles[tileIndex].capitalOf > 0? 'Capital' : 'City'}`,
                    population: 0,
                    progress: 0,
                    rewards: new Set(),
                    borderSize: 1,
                    connectedToCapital: false,
                    level: 1,
                    // 1 level + 1 capital + 1 if luxidor
                    production: 1 + 1 + (tribeType == TribeType.Luxidoor? 1 : 0),
                    owner: tribe.id,
                    _territory: territory,
                };

                state.tribes[tribe.id].cities.push(cityData);

                state.structures[tileIndex] = {
                    type: StructureType.Village,
                    level: cityData.level,
                    founded: 0,
                    score: 0,
                    tileIndex,
                }
            }
            else if (structureOrTribeType == 'rs') {
                state.structures[tileIndex] = {
                    type: StructureType.Ruin,
                    level: 0,
                    founded: 0,
                    score: 0,
                    tileIndex,
                }
            }
        }

        // Set resources

        for (let i = 0; i < resourceRaw.length; i++) {
            const pResource = resourceRaw[i];

            if (pResource != 'y') continue;

            if (state.structures[i] && state.structures[i]!.type != StructureType.Ruin) continue;

            let resourceType = ResourceType.None;

            switch (state.tiles[i].type) {
                case TerrainType.Forest:
                    resourceType = ResourceType.Game;
                    break;
                case TerrainType.Mountain:
                    resourceType = ResourceType.Metal;
                    break;
                case TerrainType.Water:
                    resourceType = ResourceType.Fish;
                    break;
                case TerrainType.Ocean:
                    resourceType = ResourceType.Starfish;
                    break;
                case TerrainType.Field:
                    resourceType = ResourceType.Fruit;
                default:
                    break;
            }

            if (!isResourceVisible(state.tribes[state.settings.currentPlayerTurnId], resourceType)) {
                (state as any)._hiddenResources[i] = resourceType;
            }
    
            state.resources[i] = {
                type: resourceType,
                tileIndex: i
            }
        }

        // Spawn starting units
        // Validate state

        for(const owner in state.tribes) {
            const tribe = state.tribes[owner];
            if (tribe.cities.length != 1) {
                throw Error(`Tribe ${TribeType[tribe.type]} has ${tribe.cities.length} cities`);
            }
            const capital = tribe.cities[0];
            state.settings.currentPlayerTurnId = tribe.id;
            summonUnit(
                state,
                TribeSettings[tribe.type].uniqueStartingUnit || UnitType.Warrior,
                capital.tileIndex,
            );
            if (tribe.id == pov) {
                tribe.units[0].moved = false;
                tribe.units[0].attacked = false;
            }
            tribe.score = calculateInitialTribeScore(state, tribe.id);
        }

        state.settings.currentPlayerTurnId = pov;

        this.loadGame(state);
    }

    public saveTo(filename: string) {
        writeFileSync(`data/${filename}.json`, JSON.stringify({
            ...this.currentState,
            hash: this.currentState.tribes[this.currentState.settings.currentPlayerTurnId]._hash.toString(),
        } as any, null, 2));
        console.log(`Saved state to data/${filename}.json`);
    }

    private updatePredictions(state: GameState) {
        const villagePredictions = predictVillages(state);
        const fogPredictions: { [tileIndex: number]: [TerrainType, ClimateType, boolean] } = {};

        const prediction: any = { };

        prediction._villages = villagePredictions;

        if (!Object.keys(villagePredictions).length) {
            prediction._villages = undefined;
        }
        else {
            Object.entries(villagePredictions).forEach(([tileIndex, tribeType]) => {
                const climateType = ClimateType[TribeType[tribeType[0]] as keyof typeof ClimateType];
                fogPredictions[Number(tileIndex)] = [
                    TerrainType.Field,
                    climateType,
                    true
                ];
                getAdjacentIndexes(state, Number(tileIndex), 1, false, true).forEach(x => {
                    fogPredictions[x] = [
                        TerrainType.Field,
                        climateType,
                        false
                    ];
                });
            });
        }

        prediction._terrain = predictOuterFogTerrain(state, fogPredictions);

        if (!Object.keys(prediction._terrain).length) {
            prediction._terrain = undefined;
        }

        prediction._enemyCapitalSuspects = undefined;//predictEnemyCapitalsAndSurroundings(state);

        prediction._cityRewards = predictBestNextCityReward(state);

        state._prediction = prediction;
    }

    /**
    * Parses a string into a boolean.
    * Returns false if the string is '0', otherwise true.
    */
    private parseRawBool(x: string): boolean {
        return x === "0" ? false : true;
    }

    /**
    * Parses a string into a number.
    * Returns -1 if the parsed number is greater than some large impossible number.
    */
    private parseRawInt(x: string, cap = true): number {
        if (x == null || x == "" || x == undefined) return -1;
        const parsed = Number.parseInt(x);
        return cap && parsed > 60000 && parsed < 70000? -1 : parsed;
    }
}
