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
import { UnitState, CityState, GameState, TribeState, TileState, ResourceState, StructureState, DiplomacyRelationState, GameSettings, DefaultGameSettings } from "./states";
import { isResourceVisible, getNeighborTiles, getNeighborIndexes, isWaterTerrain, isIceTerrain, getTribeCrudeScore, getHomeCity, cloneState } from "./functions";
import { readFileSync, writeFileSync } from "fs";
import { UnitSettings } from "./settings/UnitSettings";
import { predictBestNextCityReward, predictOuterFogTerrain, predictVillages } from "../eval/prediction";
import { TribeSettings } from "./settings/TribeSettings";
import { summonUnit } from "./actions";
import { calculateInitialZobristHash } from "../zorbist/hasher";
import { zobristKeys } from "../zorbist/zobristKeys";

export const STARTING_OWNER_ID = 1;
export const DEBUG_SEED = undefined;
export const MAX_SEED = 10;
// Standard max turns when loaded live games
export const MAX_TURNS = 50;

export default class GameLoader {
    public currentState: GameState;
    readonly fow: boolean;
    readonly settings: GameSettings;

    constructor(settings?: GameSettings) {
        this.fow = false;
        this.currentState = { } as any;
        this.settings = {
            ...DefaultGameSettings,
            seed: DEBUG_SEED,
            ...(settings || {})
        };
        if(!this.settings.mode) throw new Error("mode is required");
        if(!this.settings.maxTurns) throw new Error("maxTurns is required");
        if(!this.settings.size) throw new Error("size is required");
    }
  
    private defaultState(): GameState {
        return {
            hash: 0n,
            settings: {
                size: 0,
                _turn: 0,
                maxTurns: 0,
                _pov: 0,
                areYouSure: false,
                unitIdx: 0,
                tribeCount: 0,
                mode: ModeType.Perfection,
                _gameOver: false,
                _recentMoves: [],
                _pendingRewards: []
            },
            tiles: [],
            structures: {},
            resources: {},
            tribes: { },
            _visibleTiles: []
        }    
    }
    
    private async readLiveGameData(): Promise<string[][]> {
        return new Promise((resolve, reject) => {
            exec("bash scan.sh -y", (error: any, stdout: string, stderr: any) => {
                if(error) {
                    if(error.code == 1) {
                        return reject(stdout.trim());
                    }
                    if (!stdout.length && (error || stderr)) {
                        return reject(error || stderr);
                    }
                }
                try {
                    const data: any = stdout.trim().split("\n");
                    data[0] = data[0].split(",");
                    data[1] = data[1].split(";").filter(Boolean);
                    data[2] = data[2].split("+");
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            });
        });
    }

    private loadGame(state: GameState) {
        this.currentState = state;

        // this.updatePredictions(state);

        // If FOW is disabled, then tribes shouldn't claim discovering other tribes
        if(!this.fow) {
            const tribesObj = Object.values(state.tribes);
            for(const tribe of tribesObj) {
                tribesObj.forEach(x => {
                    if(x.owner != tribe.owner) {
                        tribe._knownPlayers.add(x.owner);
                    }
                });
            }
        }

        state.hash = calculateInitialZobristHash(this.currentState, zobristKeys);
    }

    public async loadLive() {
        const data = await this.readLiveGameData().catch((err) => {
            console.log(err);
            return null;
        });

        if(!data) return null;
        
        let [ [ size, turn ], playerStates, mapdata ] = data;

        const state: GameState = {
            ...this.defaultState(),
            settings: {
                mode: ModeType.Domination,
                size: this.parseRawInt(size),
                _turn: this.parseRawInt(turn),
                maxTurns: MAX_TURNS, // TODO
                _pov: STARTING_OWNER_ID,
                areYouSure: false,
                unitIdx: 0,
                tribeCount: 0,
                _gameOver: false,
                _recentMoves: [],
                _pendingRewards: [],
            },
            tribes: playerStates.reduce((arr, line) => {
                const [
                    owner,
                    username,
                    bot,
                    score,
                    stars,
                    techlist,
                    tribeId,
                    killerId,
                    kills,
                    tasks,
                    builtUniqueImprovements,
                    knownPlayers,
                    relations,
                    killedTurn,
	                resignedTurn,
                ] = line.split(",");
                const index = this.parseRawInt(owner);
                if (index > 20) return arr;
                return {
                    ...arr,
                    [index]: {
                        owner: index,
                        username,
                        bot: this.parseRawBool(bot),
                        tribeType: this.parseRawInt(tribeId),
                        _stars: this.parseRawInt(stars),
                        _score: this.parseRawInt(score, false),
                        _tech: techlist.split("&").map((x) => this.parseRawInt(x)).map(x => ({
                            techType: x as TechnologyType,
                            discovered: true,
                        })),
                        _killerId: this.parseRawInt(killerId),
                        _killedTurn: this.parseRawInt(killedTurn),
                        _resignedTurn: this.parseRawInt(resignedTurn),
                        _kills: this.parseRawInt(kills),
                        _casualties: 0,
                        // tasks: tasks.split("&").map((values) => {
                        //     const [started, completed, turn] = values.split("-");
                        //     return {
                        //         started: this.parseRawBool(started),
                        //         completed: this.parseRawBool(completed),
                        //         customData: this.parseRawInt(turn),
                        //     };
                        // }),
                        _builtUniqueStructures: new Set(builtUniqueImprovements.split("&").map((x) => this.parseRawInt(x))),
                        _cities: [],
                        _units: [],
                        _terrirory: [],
                        _structures: [],
                        _knownPlayers: new Set(knownPlayers.split("&").map((x) => this.parseRawInt(x))),
                        relations: relations.split("&").reduce((acc, data) => {
                            const [key, values] = data.split("_");                            
                            const [state, lastAttackTurn, embassyLevel, lastPeaceBrokenTurn, firstMeet, embassyBuildTurn, previousAttackTurn] = values.split("-");
                            return {
                                ...acc,
                                [Number(key)]: {
                                    state: this.parseRawBool(state),
                                    lastAttackTurn: this.parseRawInt(lastAttackTurn),
                                    embassyLevel: this.parseRawInt(embassyLevel),
                                    lastPeaceBrokenTurn: this.parseRawInt(lastPeaceBrokenTurn),
                                    firstMeet: this.parseRawInt(firstMeet),
                                    embassyBuildTurn: this.parseRawInt(embassyBuildTurn),
                                    previousAttackTurn: this.parseRawInt(previousAttackTurn),
                                }
                            } as DiplomacyRelationState;
                        }, {}),
                    } as TribeState,
                };
            }, {}),
            _visibleTiles: [],
        };

        const tribes = Object.values(state.tribes);
        state.settings.tribeCount = tribes.length;
        
        // ! used only for resources custom visibility
        const playerTribe = state.tribes[1]!;

        for (const segment of mapdata) {
            const values = segment.split(";").map((x) => x.split(","));
            
            if(!values[1]) continue;

            const tileIndex = this.parseRawInt(values[0][0]);

            const [ , rawTile, rawStructure, rawResource, rawUnit, city ] = values;
            
            if (!rawTile || rawTile.length === 0) break;
            
            const tileOwner: number = this.parseRawInt(rawTile[1]) < 1? -1 : this.parseRawInt(rawTile[1]);
            
            const explorers = this.fow? rawTile[2].split("&").map((x) => this.parseRawInt(x)).filter(x => x > 0) : tribes.map(x => x.owner);
            
            const tileData: TileState = {
                terrainType: this.parseRawInt(rawTile[0]),
                _owner: tileOwner,
                _explorers: new Set(explorers),
                hasRoad: this.parseRawBool(rawTile[3]),
                hasRoute: this.parseRawBool(rawTile[4]),
                hadRoute: this.parseRawBool(rawTile[5]),
                capitalOf: this.parseRawInt(rawTile[6]),
                climate: this.parseRawInt(rawTile[9]),
                skinType: this.parseRawInt(rawTile[10]),
                x: tileIndex % state.settings.size,
                y: Math.floor(tileIndex / state.settings.size),
                tileIndex,
                _rulingCityIndex: this.parseRawInt(rawTile[8]) < 0 || this.parseRawInt(rawTile[7]) < 0? -1 : (this.parseRawInt(rawTile[7]) + this.parseRawInt(rawTile[8]) * state.settings.size),
                _unitOwner: 0,
            };
            
            // ! if there is a structure
            if (rawStructure && rawStructure[0]) {
                const [ id, level, turn, reward ] = rawStructure;
                const structureData: StructureState = {
                    id: this.parseRawInt(id),
                    _level: this.parseRawInt(level),
                    turn: this.parseRawInt(turn),
                    reward: this.parseRawInt(reward),
                    tileIndex,
                };

                state.structures[tileIndex] = structureData;
            }
            
            // ! if there is a resource
            if (rawResource && rawResource[0]) {
                const resourceId = this.parseRawInt(rawResource[0]);
                const resourceData: ResourceState = {
                    id: resourceId,
                    tileIndex
                };

                state.resources[tileIndex] = resourceData;

                if(!isResourceVisible(playerTribe, resourceId)) {
                    (state as any)._hiddenResources[tileIndex] = resourceId;
                }
            }
            
            // ! unit in data not bound to tile index, first N count tile index are always units
            if (rawUnit && rawUnit[0]) {
                const effects = rawUnit[17].split("&").map((x) => this.parseRawInt(x));
                const unitData: UnitState = {
                    _tileIndex: -1,
                    _owner: this.parseRawInt(rawUnit[0]),
                    x: this.parseRawInt(rawUnit[1]),
                    y: this.parseRawInt(rawUnit[2]),
                    _unitType: this.parseRawInt(rawUnit[3]),
                    _health: this.parseRawInt(rawUnit[4]),
                    veteran: this.parseRawBool(rawUnit[5]),
                    kills: this.parseRawInt(rawUnit[6]),
                    prevX: this.parseRawInt(rawUnit[7]),
                    prevY: this.parseRawInt(rawUnit[8]),
                    _homeIndex: this.parseRawInt(rawUnit[9]) + this.parseRawInt(rawUnit[10]) * state.settings.size,
                    direction: this.parseRawInt(rawUnit[11]),
                    flipped: this.parseRawBool(rawUnit[12]),
                    createdTurn: this.parseRawInt(rawUnit[13]),
                    _moved: this.parseRawBool(rawUnit[14]),
                    _attacked: this.parseRawBool(rawUnit[15]),
                    _effects: new Set(effects),
                    _passenger: this.parseRawInt(rawUnit[16]) > 0 ? this.parseRawInt(rawUnit[16]) : undefined,
                };

                if(!UnitSettings[unitData._unitType]) {
                    throw new Error(`Unit type "${unitData._unitType}" not found!`);
                }

                unitData._tileIndex = unitData.x + unitData.y * state.settings.size;

                state.tribes[unitData._owner]._units.push(unitData);
            }
            
            if (city && city[0]) {
                // City
                if(tileOwner > 0) {
                    // ! make sure reward id is valid
                    const cityData: CityState = {
                        name: city[0],
                        _population: this.parseRawInt(city[1]),
                        _progress: this.parseRawInt(city[2]),
                        _rewards: new Set(city[3].split("&").map((x) => this.parseRawInt(x))),
                        _borderSize: this.parseRawInt(city[5]),
                        _connectedToCapital: this.parseRawBool(city[6]),
                        _level: this.parseRawInt(city[7]),
                        _production: 0, // this.parseRawInt(city[4])
                        _owner: tileOwner,
                        tileIndex,
                        _territory: [],
                        _unitCount: 0,
                    };

                    let count = 0;
                    cityData._rewards.forEach(x => {
                        if(x === RewardType.Workshop || x === RewardType.Park) {
                            count++;
                        }
                    })
    
                    cityData._production =
                        cityData._level +
                        (tileData.capitalOf > 0? 1 : 0) +
                        count;
    
                    state.tribes[cityData._owner]._cities.push(cityData);
                }
            }

            state.tiles[tileIndex] = tileData;
        }

        for(const tribeId in state.tribes) {
            const tribe = state.tribes[Number(tribeId)];
            for (let i = 0; i < tribe._units.length; i++) {
                const unit = tribe._units[i];
                const tile = state.tiles[unit._tileIndex];

                tile._unitOwner = tribe.owner;

                if(!isWaterTerrain(tile) && !isIceTerrain(tile)) {
                    unit._passenger = undefined;
                }
                else if(unit._passenger && unit._passenger < 2) {
                    unit._passenger = undefined;
                }

                // Skip units rewarded by ruins, or stray units (idk really)
                
                const city = getHomeCity(state, unit);

                if(city) city._unitCount++;
                else unit._homeIndex = -1;
            }
            for(const city of tribe._cities) {
                state.tiles[city.tileIndex]._rulingCityIndex = city.tileIndex
                city._territory = getNeighborTiles(state, city.tileIndex, city._borderSize)
                    .filter(x => x._rulingCityIndex == city.tileIndex).map(x => x.tileIndex);
            }
        }

        this.loadGame(state);
    }

    public async loadRandom(_seed?: number) {
        const randomNotation = async (seed: number) => {
            const mapdata: { type: string, tribe: string, above: string | null, road: boolean }[] = JSON.parse(await new Promise((resolve, reject) => {
                const cmd = `.venv/bin/python mapgen/main.py --seed ${seed} --size ${this.settings.size} --tribes ${this.settings.tribes.map(x => TribeType[x]).join(" ")}`
                exec(cmd, (error: any, stdout: string, stderr: any) => {
                    if(error) {
                        // console.log(error);
                        return reject(error || stderr);
                    }
                    resolve(stdout.trim());
                });
            }));
    
            // Convert mapdata to notation for simplicity
            return [
                // Settings
                [`${ModeType[this.settings.mode!].toLowerCase()},0,${this.settings.maxTurns},1`],
                // Tribes
                this.settings.tribes.map(x => TribeType[x].slice(0, 2).toLowerCase()),
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

        // Safeguard for inconsistent map generation
        let tries = 100
        let seed = _seed || (this.settings.seed? this.settings.seed : Math.floor(Math.random() * MAX_SEED));

        while(tries > 0) {
            try {
                const not = await randomNotation(seed).catch(() => null);
                if(!not) throw 'err';
                this.loadNotation(not);
                console.log('SEED', seed);
                return;
            } catch (error) {
                console.log(error);
                tries--;
                seed++;
                // console.log(error);
            }
        }

        console.log(`TRIED ${100} TIMES AND ALL FAILED! USING EMERGENCY STATE!`);
    }

    public loadSave(filename: string) {
        const state = JSON.parse(readFileSync(`data/${filename}.json`, 'utf-8')) as GameState;
        this.loadGame(state);
    }   

    public loadNotation(notation: string) {
        const [settingsRaw, tribeRaw, climateRaw, terrainRaw, resourceRaw, structuresRaw] = notation.split(';');

        // Settings (mode, turn, maxturn, pov)
        const settings = settingsRaw.split(',');

        const pov = Number(settings[3]);

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
                    owner,
                    username: owner == pov? "Player" : TribeType[type],
                    bot: owner != pov,
                    tribeType: type,
                    _score: 0,
                    _stars: 5,
                    _killedTurn: -1,
                    _resignedTurn: -1,
                    _killerId: -1,
                    _tech: [TechnologyType.None, ...TribeSettings[type].startingTech? [TribeSettings[type].startingTech] : []].map(x => ({
                        techType: x as TechnologyType,
                        discovered: true,
                    })),
                    _kills: 0,
                    _casualties: 0,
                    tasks: [],
                    _builtUniqueStructures: new Set(),
                    _cities: [],
                    _units: [],
                    _resources: [],
                    _structures: [],
                    _knownPlayers: new Set(),
                    relations: [],
                } as TribeState,
            };
        }, {}) as { [key: number]: TribeState };

        const state: GameState = {
            ...this.defaultState(),
            settings: {
                mode: settings[0] == 'domination'? ModeType.Domination : ModeType.Perfection,
                size: Math.sqrt(Number(climateRaw.length / 2)),
                _turn: Number(settings[1]),
                maxTurns: Number(settings[2]),
                _pov: pov,
                areYouSure: false,
                unitIdx: 0,
                tribeCount: Object.keys(tribes).length,
                _gameOver: false,
                _recentMoves: [],
                _pendingRewards: [],
            },
            tribes
        };

        for(const owner in state.tribes) {
            state.tribes[owner].relations = Object.values(state.tribes).reduce((acc: any, tribe) => ({
                ...acc,
                [tribe.owner]: {
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

        // ! Always assuming this is turn 0

        const lighthouses = [
            0,
            state.settings.size - 1,
            state.settings.size * state.settings.size - 1,
            1 + state.settings.size * state.settings.size - state.settings.size
        ];

        // Set tiles

        const climateTypes = climateRaw.match(/.{1,2}/g)!;
        const terrainTypes = terrainRaw.match(/./g)!;
        const owners = this.fow? [] : Object.values(tribes).map(x => x.owner);
        for(let i = 0; i < climateTypes.length; i++) {
            const climate = TribeMap[climateTypes[i]]? ClimateType[TribeType[TribeMap[climateTypes[i]]] as any] as unknown as ClimateType : ClimateType.Nature;
            
            state.tiles[i] = {
                _owner: -1,
                tileIndex: i,
                climate,
                // If tile is ocean tile, then its nature??
                terrainType: TerrainMap[terrainTypes[i]],
                _explorers: new Set(owners),
                hasRoad: false,
                hasRoute: false,
                hadRoute: false,
                capitalOf: -1,
                skinType: -1,
                x: i % state.settings.size,
                y: Math.floor(i / state.settings.size),
                _rulingCityIndex: -1,
                _unitOwner: 0,
            }
        }

        // Set spawning cities, ruins and starfish

        for(let i = 0; i < structuresRaw.length; i += 2) {
            const tileIndex = i / 2;
            const structureOrTribeType = structuresRaw.substring(i, i + 2);

            if(structureOrTribeType == 'vv') {
                state.structures[tileIndex] = {
                    id: StructureType.Village,
                    _level: 1,
                    turn: 0,
                    reward: 0,
                    tileIndex,
                }
            }
            else if(TribeMap[structureOrTribeType]) {
                const tribeType = TribeMap[structureOrTribeType];
                const territory = [tileIndex, ...getNeighborIndexes(state, tileIndex, 1, false, true)];
                // const tribe = Object.values(state.tribes).find(x => x.tribeType == tribeType)!;
                const tribe = Object.values(state.tribes)
                    .filter(x => x.tribeType === tribeType)
                    .find(x => x._cities.length === 0)!;

                for(const tile of territory) {
                    state.tiles[tile] = {
                        ...state.tiles[tile],
                        _owner: tribe.owner,
                        capitalOf: tribe.owner,
                        _rulingCityIndex: tileIndex,
                    }
                }
                
                // Reveal surrounding land
                if(this.fow) {
                    for(const tile of [
                        tileIndex, 
                        ...getNeighborIndexes(state, tileIndex, 2, false, true).filter(x => !lighthouses.includes(x))
                    ]) {
                        state.tiles[tile]._explorers.add(tribe.owner);
                    }
                }

                const cityData: CityState = {
                    name: `${TribeType[tribeType]} ${state.tiles[tileIndex].capitalOf > 0? 'Capital' : 'City'}`,
                    _population: 0,
                    _progress: 0,
                    _rewards: new Set(),
                    _borderSize: 1,
                    _connectedToCapital: false,
                    _level: 1,
                    // 1 level + 1 capital + 1 if luxidor
                    _production: 1 + 1 + (tribeType == TribeType.Luxidoor? 1 : 0),
                    _owner: tribe.owner,
                    tileIndex,
                    _territory: territory,
                    _unitCount: 0,
                };

                state.tribes[tribe.owner]._cities.push(cityData);
                
                state.structures[tileIndex] = {
                    id: StructureType.Village,
                    _level: cityData._level,
                    turn: 0,
                    reward: 0,
                    tileIndex,
                }
            }
            else if(structureOrTribeType == 'rs') {
                state.structures[tileIndex] = {
                    id: StructureType.Ruin,
                    _level: 0,
                    turn: 0,
                    reward: 0,
                    tileIndex,
                }
            }
        }

        // Set resources

        (state as any)._hiddenResources = [];

        for (let i = 0; i < resourceRaw.length; i++) {
            const pResource = resourceRaw[i];
            
            if(pResource != 'y') continue;
            
            if(state.structures[i] && state.structures[i]!.id != StructureType.Ruin) continue;

            let resourceType = ResourceType.None;

            switch (state.tiles[i].terrainType) {
                case TerrainType.Forest:
                    resourceType = ResourceType.WildAnimal;
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
            
            if(!isResourceVisible(state.tribes[state.settings._pov], resourceType)) {
                (state as any)._hiddenResources[i] = resourceType;
                continue;
            }

            state.resources[i] = {
                id: resourceType,
                tileIndex: i
            }
        }

        // Spawn starting units
        // Validate state

        for(const owner in state.tribes) {
            const tribe = state.tribes[owner];
            if(tribe._cities.length != 1) throw Error(`Tribe ${TribeType[tribe.tribeType]} has ${tribe._cities.length} cities`);
            state.settings._pov = tribe.owner;
            tribe._cities[0]._unitCount = 1;
            summonUnit(
                state, 
                TribeSettings[tribe.tribeType].uniqueStartingUnit || UnitType.Warrior, 
                tribe._cities[0].tileIndex,
            );
            if(tribe.owner == pov) {
                tribe._units[0]._moved = false;
                tribe._units[0]._attacked = false;
            }
            tribe._score = getTribeCrudeScore(state, tribe.owner);
        }

        state.settings._pov = pov;

        this.loadGame(state);
    }

    public saveTo(filename: string) {
        writeFileSync(`data/${filename}.json`, JSON.stringify({ 
            ...this.currentState,
            hash: this.currentState.hash.toString(), 
        } as any, null, 2));
        console.log(`Saved state to data/${filename}.json`);
    }

    private updatePredictions(state: GameState) {
        const villagePredictions = predictVillages(state);
        const fogPredictions: { [tileIndex: number]: [TerrainType, ClimateType, boolean] } = {};

        const prediction: any = { };

        prediction._villages = villagePredictions;
    
        if(!Object.keys(villagePredictions).length) {
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
                getNeighborIndexes(state, Number(tileIndex), 1, false, true).forEach(x => {
                    fogPredictions[x] = [
                        TerrainType.Field, 
                        climateType,
                        false
                    ];
                });
            });
        }
        
        prediction._terrain = predictOuterFogTerrain(state, fogPredictions);

        if(!Object.keys(prediction._terrain).length) {
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
        if(x == null || x == "" || x == undefined) return -1;
        const parsed = Number.parseInt(x);
        return cap && parsed > 60000 && parsed < 70000? -1 : parsed;
    }
}
