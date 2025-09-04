const CLIMATE_IDS = [
    "Nature", "Xin-xi", "Imperius", "Bardur", "Oumaji",
    "Kickoo", "Hoodrick", "Luxidoor", "Vengir",
    "Zebasi", "Ai-Mo", "Aquarion", "Quetzali",
    "∑∫ỹriȱŋ", "Yădakk", "Polaris", "Cymanti"
];

const TRIBE_IDS = [
    null, 'Nature', 'Ai-Mo', 'Aquarion', 'Bardur', 
    '∑∫ỹriȱŋ', 'Hoodrick', 'Imperius', 'Kickoo', 
    'Luxidoor', 'Oumaji', 'Quetzali', 'Vengir', 
    'Xin-xi', 'Yădakk', 'Zebasi', 'Polaris', 'Cymanti'
];

const ClassNameToId = {
    1: "Scout",
    2: "Warrior",
    3: "Rider",
    4: "Knight",
    5: "Defender",
    7: "Battleship",
    8: "Catapult",
    9: "Archer",
    10: "MindBender",
    11: "Swordsman",
    12: "Giant",
    15: "Polytaur",
    20: "Amphibian",
    21: "Tridention",
    22: `Mooni`,
    23: "BattleSled",
    25: "IceArcher",
    26: "Crab",
    28: "Hexapod",
    31: "Kiton",
    35: "Raychi",
    36: "Shaman",
    38: "Cloak",
    39: "Cloak_Boat",
    41: "Bombership",
    42: "Scoutship",
    43: "Boat",
    44: "Rammership",
    45: "Juggernaut",
};

const CLIMATE_TO_ANIMAL = [
    'Invalid', 
    'horse0001', // xinxi,
    'horse0002', // imperius,
    'horse0003', // bardur,
    'horse0004', // oumaji,
    'horse0005', // kickoo,
    'horse0006', // hoodrick,
    'horse0007', // luxidoor,
    'horse0008', // vengir,
    'horse0009', // zebasi,
    'horse0010', // aimo,
    'horse0011', // aquarion,
    'horse0012', // quetzali,
    'horse0013', // '∑∫ỹriȱŋ',
    'horse0014', // yadakk,
    'animal_15', // polaris,
    'lytheti', // cymanti
]

const OWNER_TO_ID_INDEX = [
    null, 0, 10, 11, 3, 13, 6, 2, 5, 7, 4, 12, 8, 1, 14, 9, 15, 16
]

var state = {};
var TILE_ELEMENTS = {};
var TILE_FOW = [];
var FOG_OF_WAR = true;
var unitMovePreviewTiles = [];
var unitAttackPreviewTiles = [];
const tileSize = 128;
const someOffset = 4; // idk
var POV = 1;
const mapContainer = document.getElementById("map");

const TerrainType = {
	0: "None",
	1: "Water",
	2: "Ocean",
	3: "Land",
	4: "Mountain",
	5: "Forest",
	6: "Ice",
	7: "GroundWater",
}

function getNeighborTiles(x, y, range = 2) {
    const neighbors = [];
    for (let dx = -range; dx <= range; dx++) {
        for (let dy = -range; dy <= range; dy++) {
            if (dx !== 0 && dy !== 0) {
                neighbors.push({ x: x + dx, y: y + dy });
            }
        }
    }
    return neighbors;
}

function rotate1DArray90(arr, size) {
    // Convert 1D array to 2D matrix
    let matrix = [];
    for (let i = 0; i < size; i++) {
        matrix.push(arr.slice(i * size, (i + 1) * size));
    }
    
    // Rotate the matrix 90 degrees clockwise
    let rotated = Array.from({ length: size }, () => Array(size).fill(0));
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            rotated[j][size - 1 - i] = matrix[i][j];
        }
    }
    
    // Flatten rotated 2D array back into 1D array
    return rotated.flat();
}

function generateMap(newState = null) {
    if(newState) {
        state = newState;
    }

    document.getElementById('turn').textContent = `${state.settings._turn} / ${state.settings.maxTurns}`;
    document.getElementById('stars').textContent = state.tribes[state.settings._pov]._stars;
    document.getElementById('production').textContent = state.tribes[state.settings._pov]._cities.reduce((acc, cur) => acc + cur._production, 0);
    document.getElementById('score').textContent = state.tribes[state.settings._pov]._score || 69;

    const mapSize = state.settings.size;
   
    for(const tiles of Object.values(TILE_ELEMENTS)) {
        for(const element of tiles) {
            element.parentNode.removeChild(element);
        }
    }

    TILE_ELEMENTS = {};
        
    function createPredictionGroundTile(tileIndex, x, y, terrainType, tileClimate) {
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][terrainType] || `terrain/tiles/ground_${tileClimate}`;
        const groundTile = createTile(x, y, tilefile);
        groundTile.classList.add("tile");
        groundTile.classList.add('ignoremouse');
        groundTile.classList.add('ground');
        groundTile.style.backgroundImage = `url('textures/${tilefile}.png')`;
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);
        groundTile.style.left = `${posX}px`;
        groundTile.style.top = `${posY}px`;
        groundTile.style.zIndex = posY;
        groundTile.style.opacity = 0.85;
        mapContainer.appendChild(groundTile);
        for(const element of TILE_FOW) {
            try {
                if(tileIndex == Number(element.id.split('-')[1])) {
                    element.parentNode.removeChild(element);
                }
            } catch (error) {
                
            }
        }
    }

    function createTile(x, y, filename, z=0, offset=[0,0]) {
        const id = `${x},${y}`;
        if(!TILE_ELEMENTS[id]) TILE_ELEMENTS[id] = [];

        const tile = document.createElement("div");
        tile.classList.add("tile");
        tile.classList.add('ignoremouse');
        
        tile.style.backgroundImage = `url('textures/${filename}.png')`;
        
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);
        
        tile.style.left = `${posX}px`;
        tile.style.top = `${posY}px`;
        
        mapContainer.appendChild(tile);
        
        tile.style.zIndex = posY + z;

        TILE_ELEMENTS[id].push(tile);
        
        return tile;
    }
    
    const mapLength = mapSize * mapSize;
    
    let tiledata = state['tiles'];
    tiledata = rotate1DArray90(Object.keys(state['tiles']).reduce((arr, index, i) => [...arr, { ...tiledata[index], index }], []), mapSize);

    const suspects = state._prediction?._enemyCapitalSuspects || [];
    
    state['cities'] = Object.values(state.tribes).reduce((arr, tribe, i) => ({...arr, ...(tribe._cities.reduce((a, b) => ({...a, [b.tileIndex]: b}), {}))}), {});
    state['units'] = Object.values(state.tribes).reduce((arr, tribe, i) => ({...arr, ...(tribe._units.reduce((a, b) => ({...a, [b._tileIndex]: b}), {}))}), {});

    for (let index = 0; index < mapLength; index++) {
        const x = index % mapSize;
        const y = Math.floor(index / mapSize);
        const newIndex = (mapSize - 1 - y) * mapSize + x;

        /** @type { { tileIndex: number, _rulingCityIndex: number, terrainType: number, _owner: number, explorers: number[], hasRoad: boolean, hasRoute: boolean, hadRoute: boolean, capitalOf: number, skinType: number, climate: number } } */
        const tile = tiledata[newIndex];
        
        if(!tile) {
            console.log(`Missing tile: ${newIndex}`);
            continue;
        }

        /** @type { { _index: number, id: number } } */
        let resource = state._hiddenResources[tile.tileIndex]? {} : state['resources'][tile.tileIndex] || (state['_hiddenResources'] || {})[tile.tileIndex];

        /** @type { { _index: number, id: number, _level: number, _turn: number, reward: number } } */
        const structure = state['structures'][tile.tileIndex];
        
        /** @type { { _index: number, name: string, _population: number, _progress: number, _owner: number, _production: number } } */
        const city = state['cities'][tile.tileIndex];
        
        /** @type { { _index: number, _owner: number, _unitType: number, x: number, y: number, _health: number, _kills: number, class: { id: number, health: number, defense: number, movement: number, attack: number, cost: number } } } */
        const unit = state['units'][tile.tileIndex];
        
        const tileTribeName = CLIMATE_IDS[tile.climate];
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][tile.terrainType] || `terrain/tiles/ground_${tile.climate}`;
        const groundTile = createTile(x, y, tilefile);
        groundTile.dataset.tileIndex = tile.tileIndex;

        groundTile.id = `tile,${x},${y}`;
        groundTile.classList.add('ground');
        groundTile.classList.remove('ignoremouse');

        if(suspects.includes(tile.tileIndex)) {
            groundTile.style.filter = 'grayscale(1)';
        }
        
        const ambientfile = { 
            4: `terrain/mountains/mountain_${tile.climate}`, 
            5: `terrain/forests/Forest_${tile.climate}` 
        }[tile.terrainType];
        
        if(ambientfile) {
            if(tile.terrainType == 4) createTile(x, y, ambientfile, 3).classList.add('mountain');
            else createTile(x, y, ambientfile, 1).classList.add('forest');
        }
        
        if(city) {
            const tribeNameIndex = CLIMATE_IDS.indexOf(TRIBE_IDS[state.tribes[city._owner].tribeType]);
            const cityTribeName = CLIMATE_IDS[tribeNameIndex];
            const e = createTile(x, y, `buildings/${cityTribeName}/Default/Houses/House_${tribeNameIndex}_5`, 999);
            e.classList.add('city');
            e.id = `city-${city.name.toLowerCase()}`;
            e.appendChild(document.createElement('div')).innerHTML = `
            <p>
            <span class="${tile.capitalOf > 0? 'capital' : ''}">${city.name}</span> (+${city._production})<br>
            ${city._progress} / ${city._level + 1} (${city._population}) [${city._unitCount}]<br>
            </p>
            `;
        }
        else if(structure) {
            const structId = Number(structure.id);
            const structureFile = { 
                1: `buildings/common/Tribe`, 
                2: `terrain/misc/ResourceGFX_ruin`, 
                5: `buildings/common/Farm`, 
                6: `buildings/common/Windmill_${structure._level}`, // TODO might be x-1
                8: `buildings/common/Port`, 
                12: `buildings/common/Lumber Hut`, 
                13: `buildings/common/Sawmill_${structure._level}`,
                17: `buildings/common/Temple_${structure._level}`, 
                18: `buildings/common/Water Temple_${structure._level}`, 
                19: `buildings/common/Forest Temple_${structure._level}`, 
                20: `buildings/common/Mountain Temple_${structure._level}`, 
                21: `buildings/common/Mine`, 
                22: `buildings/common/Forge_${structure._level}`, 
                23: `buildings/${tileTribeName}/Default/Monuments/Monument1_${tile.climate}`, // altar of peace
                24: `buildings/${tileTribeName}/Default/Monuments/Monument2_${tile.climate}`, // tower_of_wisdom
                25: `buildings/${tileTribeName}/Default/Monuments/Monument3_${tile.climate}`, // grand_bazaar
                26: `buildings/${tileTribeName}/Default/Monuments/Monument4_${tile.climate}`, // emperors_tomb
                27: `buildings/${tileTribeName}/Default/Monuments/Monument5_${tile.climate}`, // gate_of_power
                28: `buildings/${tileTribeName}/Default/Monuments/Monument6_${tile.climate}`, // park_of_fortune
                29: `buildings/${tileTribeName}/Default/Monuments/Monument7_${tile.climate}`, // eye_of_god
                32: `misc/missing`, // TODO
                33: `buildings/Polaris/Default/Unique/iceport`,
                37: `buildings/Cymanti/Default/Unique/spores_${structure._level}`, 
                38: `buildings/Cymanti/Default/Unique/swamp`,
                39: `buildings/Cymanti/Default/Unique/Mycelium_${structure._level}`, 
                47: `misc/missing`, // TODO
                48: `buildings/common/bridge`,
                50: `buildings/common/Market01`,
                69: `buildings/common/Ice Temple_${structure._level}`, 
            }[structId];
            if(structureFile) {
                if(structureFile == "misc/missing") {
                    console.log("MISSING STRUCTURE:", `${structId}${structId == 47? ' (lighthouse)' : ''}`);
                }   
                // ? disabled cause its annoying
                if(structId != 47) {
                    const e = createTile(x, y, structureFile, 3);
                    if(structId > 16 && structId < 21) {
                        e.classList.add('temple');
                    }
                    else {
                        switch (structId) {
                            case 1: e.classList.add('village'); break;
                            case 2: e.classList.add('ruins'); break;
                            case 12: e.classList.add('lumberhut'); break;
                            case 21: e.classList.add('mine'); break;
                            case 22: e.classList.add('forge'); break;
                            case 33: e.classList.add('iceport'); break;
                        }
                    }
                    e.classList.add('structure');
                    e.id = `struct-${structId}`;
                }
            }
            else if(structId) {
                throw new Error(`MISSING STRUCT: ${structId}`)
            }
        }
        else if(resource) {
            const resourceFile = { 
                1: `animals/${CLIMATE_TO_ANIMAL[tile.climate]}`,
                2: `terrain/misc/ResourceGFX_crop`,
                3: `animals/fish`,
                5: `terrain/misc/ResourceGFX_metal`,
                6: `fruits/ResourceGFX_fruit_${tile.climate}`,
                7: `fruits/ResourceGFX_fruit_16`, // ? spores
                8: `terrain/misc/ResourceGFX_starfish`,
            }[resource.id];
            if(resourceFile) {
                if(resourceFile == "misc/missing") {
                    console.log("MISSING RESOURCE:", resource.id);
                }
                const e = createTile(x, y, resourceFile, 3);
                e.id = `resource-${resource.id}-${tile.index}`;
                if(resource.id > 16 && resource.id < 21) {
                    e.classList.add('temple');
                }
                else {
                    switch (resource.id) {
                        case 1: e.classList.add('animal'); break;
                        case 2: e.classList.add('crop'); break;
                        case 3: e.classList.add('fish'); break;
                        case 5: e.classList.add('metal'); break;
                        case 6: e.classList.add('fruit'); break;
                        case 8: e.classList.add('starfish'); break;
                    }
                }
                e.classList.add('resource');
            }
            else if(resource.id) {
                throw new Error(`MISSING RESOURCE: ${resource.id}`)
            }
        }
        
        if(unit) {
            const tribeNameIndex = CLIMATE_IDS.indexOf(TRIBE_IDS[state.tribes[unit._owner].tribeType]);
            const unitTribeName = CLIMATE_IDS[tribeNameIndex];
            const className = ClassNameToId[unit._unitType];
            if(!className) {
                throw new Error(`MISSING UNIT: ${unit._unitType}`)
            }
            else if(className == "misc/missing") {
                console.log("MISSING UNIT:", unit);
            }
            const e = createTile(x, y, `units/${unitTribeName}/default/${unitTribeName}_default_${className}`, 9999);
            e.classList.add('unit');
            if(unit._moved || unit._attacked) e.classList.add('exausted');
            if(unit.flipped) e.classList.add('flipped');
            e.id = `unit-${unit._unitType}-${index}`;
            e.appendChild(document.createElement('div')).innerHTML = `
            <p class="health">${Math.floor(unit._health/10)}</p>
            `;
            if(structure?.id == 1 && structure?._owner < 1 && structure?._owner != unit.owner && 
                (unit.prevX == unit.x && unit.prevY == unit.y || !unit._moved && !unit._attacked)
            ) {
                const hint = document.createElement('img');
                hint.classList.add('hint');
                hint.src = 'textures/misc/hint.png';
                e.appendChild(hint);
                const capturing = document.createElement('img');
                capturing.classList.add('capturing');
                capturing.src = 'textures/misc/swords.png';
                e.appendChild(capturing);
            }
        }
    }

    generateFOW();

    return;
    
    state._prediction._villages &&  Object.keys(state._prediction._villages).forEach(tileIndex => {
        let x = tileIndex % mapSize;
        let y = Math.floor(tileIndex / mapSize);
        const tileData = tiledata[(mapSize - 1 - y) * mapSize + x];
        x = tileData.tileIndex % mapSize;
        y = Math.floor(tileData.tileIndex / mapSize);

        const tribeType = state._prediction._villages[tileIndex][0];

        const tile = document.createElement("div");
        
        // Is enemy city
        if(state._prediction._villages[tileIndex][1]) {
            const tribeNameIndex = CLIMATE_IDS.indexOf(TRIBE_IDS[tribeType]);
            const cityTribeName = CLIMATE_IDS[tribeNameIndex];
            tile.classList.add('city');
            tile.style.backgroundImage = `url('textures/buildings/${cityTribeName}/Default/Houses/House_${tribeNameIndex}_5.png')`;
        }
        else {
            tile.classList.add('village');
            tile.style.backgroundImage = `url('textures/buildings/common/Tribe.png')`;
        }
        
        tile.classList.add("tile");
        tile.classList.add('ignoremouse');
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);
        tile.style.left = `${posX}px`;
        tile.style.top = `${posY}px`;
        tile.style.zIndex = 1000;
        tile.style.opacity = 0.8;
        mapContainer.appendChild(tile);

        createPredictionGroundTile(tileData.tileIndex, x, y, 3, tribeType);

        for(const element of TILE_FOW) {
            try {
                if(tileData.tileIndex == Number(element.id.split('-')[1])) {
                    element.parentNode.removeChild(element);
                }
            } catch (error) {
                
            }
        }
    });

    state._prediction._terrain && Object.keys(state._prediction._terrain).forEach(tileIndex => {
        const [terrainType, terrainClimate] = state._prediction._terrain[tileIndex];
        let x = tileIndex % mapSize;
        let y = Math.floor(tileIndex / mapSize);
        const tileData = tiledata[(mapSize - 1 - y) * mapSize + x];
        x = tileData.tileIndex % mapSize;
        y = Math.floor(tileData.tileIndex / mapSize);
        
        createPredictionGroundTile(tileData.tileIndex, x, y, terrainType, terrainClimate);

        const ambientfile = { 
            4: `terrain/mountains/mountain_${terrainClimate}`, 
            5: `terrain/forests/Forest_${terrainClimate}` 
        }[terrainType];

        if(!ambientfile) return;
        
        const tile = document.createElement("div");
        tile.classList.add("tile");
        tile.classList.add('ignoremouse');
        tile.style.backgroundImage = `url('textures/${ambientfile}.png')`;
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);
        tile.style.left = `${posX}px`;
        tile.style.top = `${posY}px`;
        tile.style.zIndex = 1000;
        tile.style.opacity = 0.5;
        mapContainer.appendChild(tile);

        if(ambientfile) {
            if(terrainType == 4) tile.classList.add('mountain');
            else tile.classList.add('forest');
        }
    });
}

function generateFOW() {
    for(const element of TILE_FOW) {
        try {
            element.parentNode.removeChild(element);
        } catch (error) {
        }
    }
    
    TILE_FOW = [];

    if(!FOG_OF_WAR) return;

    const vals = Object.values(TILE_ELEMENTS);
    const mapSize = state.settings.size;
    
    const suspects = state._prediction?._enemyCapitalSuspects || [];
    const terrainPredictions = state._prediction?._terrain? Object.keys(state._prediction._terrain) : [];

    for (let index = 0; index < vals.length; index++) {
        const tiles = vals[index];
        if(state.tiles[tiles[0].dataset.tileIndex]._explorers.includes(POV)) {
            for(const tile of tiles) {
                tile.style.visibility = 'visible';
            }
            continue;
        }
        for(const tile of tiles) {
            tile.style.visibility = 'hidden';
        }
        
        if(terrainPredictions.includes(index)) {
            continue;
        }

        const tile = document.createElement("div");
        tile.classList.add("tile");
        tile.style.backgroundImage = `url('textures/terrain/tiles/undiscovered.png')`;
        if(suspects?.includes(Number(tiles[0].dataset.tileIndex))) {
            tile.style.filter = 'hue-rotate(170deg)';
        }
        const x = index % mapSize;
        const y = Math.floor(index / mapSize);
        const posX = (x - y) * (tileSize / 2 - someOffset);
        const posY = (x + y) * (tileSize / 4 + someOffset);
        tile.style.left = `${posX}px`;
        tile.style.top = `${posY}px`;
        mapContainer.appendChild(tile);
        tile.style.zIndex = -10;
        tile.id = `fow-${index}`;
        TILE_FOW.push(tile);
    }
}

function setupDragAndZoom() {
    let scale = 1;
    let translateX = 0;
    let translateY = 0;
    let isDragging = false;
    let startX, startY;

    const mapContainer = document.getElementById("map");
    
    function updateTransform() {
        mapContainer.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;
    }
    
    mapContainer.addEventListener('wheel', function(e) {
        e.preventDefault();
        // Determine zoom direction
        const zoomIntensity = 0.1;
        let delta = -e.deltaY; // Negative deltaY means scrolling down, so invert
        // Calculate the new scale, clamped between 0.5 and 3 (adjust as needed)
        const newScale = Math.min(3, Math.max(0.5, scale + (delta > 0 ? zoomIntensity : -zoomIntensity)));
        
        // Optionally, zoom towards the mouse pointer:
        // Calculate the mouse's position relative to the container
        const rect = mapContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // Adjust translateX and translateY so that the zoom is centered on the mouse.
        // The math below adjusts the translation by the change in scale
        translateX -= (mouseX / scale - mouseX / newScale);
        translateY -= (mouseY / scale - mouseY / newScale);
        
        scale = newScale;
        updateTransform();
    });
    
    mapContainer.addEventListener('mousedown', function(e) {
        isDragging = true;
        
        startX = e.clientX;
        startY = e.clientY;
        mapContainer.classList.add('dragging');
        if(e.target.id.includes('tile,')) {
        const [, x, y] = e.target.id.split(',');
        // const city = getCityAtTile(parseInt(x), parseInt(y));
            // if(city) {
            //     const neighbors = getNeighborTiles(parseInt(x), parseInt(y), city.borderSize);
            //     for(const neighbor of neighbors) {
            //         const tile = document.getElementById(`tile,${neighbor.x},${neighbor.y}`);
            //         if(!tile) continue;
            //         tile.style.filter = 'brightness(1.5)';
            //         setTimeout(() => {
            //             tile.style.filter = '';
            //         }, 3000);
            //     }
            // }
            clickedOn(parseInt(x), parseInt(y));
        }
    });
    
    document.addEventListener('mousemove', function(e) {
        if (!isDragging) {
            if(e.target.id.includes('tile,')) {
                const hovertile = document.getElementById('hovertile');
                let [, x, y] = e.target.id.split(',');
                x = parseInt(x);
                y = parseInt(y);
                const tileElement = document.getElementById(`tile,${x},${y}`);
                if(!tileElement) return;
                const tilesData = rotate1DArray90(Object.keys(state.tiles).reduce((arr, index, i) => [...arr, { ...state.tiles[index], index }], []), state.settings.size);
                const tileData = tilesData[(state.settings.size - 1 - y) * state.settings.size + x];
                if(tileData) {
                    const anyResource = state.resources[tileData.tileIndex];
                    const anyUnit = Object.values(state.tribes).find(x => x._units.find(y => y._tileIndex == tileData.tileIndex))?._units.find(x => x._tileIndex == tileData.tileIndex);
                    if(anyUnit) {
                        hovertile.innerHTML = `
                        index: ${tileData.tileIndex}<br>
                        health: ${anyUnit._health}<br>
                        `;
                    }
                    else {
                           hovertile.innerHTML = `
                        index: ${tileData.tileIndex}<br>
                        city index: ${tileData._rulingCityIndex}<br>
                        climate: ${tileData.climate}<br>
                        owner: ${tileData._owner}<br>
                        ${anyResource?`type: ${anyResource.id}<br>`:''}
                        `;
                    }
                }
            }
            return;
        }
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        translateX += dx;
        translateY += dy;
        startX = e.clientX;
        startY = e.clientY;
        updateTransform();
    });
    
    document.addEventListener('mouseup', function(e) {
        isDragging = false;
        mapContainer.classList.remove('dragging');
    });

    document.addEventListener('mouseleave', function(e) {
        isDragging = false;
        mapContainer.classList.remove('dragging');
    });

    // ? center the map
    translateY = -200;
    updateTransform();
}

function clickedOn(x, y) {
    const tileElement = document.getElementById(`tile,${x},${y}`);
    if(!tileElement) return;
    // const mapContainer = document.getElementById("map");

    const tilesData = rotate1DArray90(Object.keys(state.tiles).reduce((arr, index, i) => [...arr, { ...state.tiles[index], index }], []), state.settings.size);
    const tileData = tilesData[(state.settings.size - 1 - y) * state.settings.size + x];
    const anyUnit = Object.values(state.tribes).find(x => x._units.find(y => y._tileIndex == tileData.tileIndex))?._units.find(x => x._tileIndex == tileData.tileIndex);

    for(const tile of unitMovePreviewTiles) {
        tile.classList.remove('preview');
    }

    if(anyUnit) {
        if(tileData._owner > 0) {
            state.settings._pov = tileData._owner;
        }
        fetch('/moves', { method: 'POST', body: JSON.stringify({ state: state, unit: anyUnit }), headers: { 'Content-Type': 'application/json' } }).then(x => x.json()).then(moves => {
            const moveCoords = moves.filter(x => x.id.includes('move'))
                .map(move => move.id.split('-').map(v => Number(v)).filter(v => !Number.isNaN(v)))
                .map(data => {
                    return [-anyUnit.y + Math.floor(data[2] / state.settings.size), anyUnit.x - (data[2] % state.settings.size)];
                });
            
            const attackCoords = moves.filter(x => x.id.includes('attack'))
                .map(move => move.id.split('-').map(v => Number(v)).filter(v => !Number.isNaN(v)))
                .map(data => {
                    return [-anyUnit.y + Math.floor(data[5] / state.settings.size), anyUnit.x - (data[5] % state.settings.size)];
                });

            for (let i = 0; i < moveCoords.length; i++) {
                const offset = moveCoords[i];
                const tileElement = document.getElementById(`tile,${x-offset[0]},${y+offset[1]}`);
                tileElement.classList.add('bright');
                setTimeout(() => {
                    tileElement?.classList.remove('bright');
                }, 10000);
            }

            for (let i = 0; i < attackCoords.length; i++) {
                const offset = attackCoords[i];
                const tileElement = document.getElementById(`tile,${x-offset[0]},${y+offset[1]}`);
                tileElement.style.filter = `hue-rotate(90deg) contrast(1.1)`;
            }
        }).then(() => {
            if(tileData._owner > 0) {
                state.settings._pov = 1;
            }
        });
    }
}

function changePov(element) {
    POV++;
    if(POV > Object.keys(state.tribes).length) POV = 1;
    element.textContent = `${TRIBE_IDS[state.tribes[POV].tribeType]}`;
    generateFOW();
}

async function autoStep(e) {
    e.disabled = true;
    const result = await fetch('/autostep', { method: 'POST', body: JSON.stringify({
        state, 
        mcts: true,
        iterations: 100,
        temperature: 0,
    }), headers: { 'Content-Type': 'application/json' } }).then(x => x.json());
    e.disabled = false;
    if(result.error) {
        return alert(result.error);
    }
    if(result.moves[0] != 'end turn') {
        for(const str of result.moves) {
            console.log(str);
        }
        if(result.reward || result.potential) {
            console.log(Number(result.reward.toFixed(3)), Number(result.potential.toFixed(3)), Math.round(result.value * 100) / 100);
        }
    }
    else {
        const povTribe = result.state.tribes[result.state.settings._pov];
        console.log('\n' + TRIBE_IDS[povTribe.tribeType].toLowerCase(), povTribe._stars);
    }
    generateMap(result.state);
}

async function mcts(e) {
    e.disabled = true;
    const result = await fetch('/mcts', { method: 'POST', body: JSON.stringify({
        state, 
        mcts: true,
        iterations: 1000,
        temperature: 0.8,
    }), headers: { 'Content-Type': 'application/json' } }).then(x => x.json());
    console.log(result.move);
    console.log(result.probs);
    e.disabled = false;
}

fetch('/random?fow=false&size=9&tribes=Imperius,Imperius').then(x => x.json()).then(x => {
// fetch('/live?fow=false').then(x => x.json()).then(x => {
    state = x.state;

    FOG_OF_WAR = state.settings.fow;

    POV = state.settings._pov;
    
    generateMap();
    
    setupDragAndZoom();

    return;

    fetch('/moves', { method: 'POST', body: JSON.stringify({ state: state, pov: 1 }), headers: { 'Content-Type': 'application/json' } }).then(x => x.json()).then(x => {
        MOVES = x;
        const map = {
            tech: [],
            units: {},
            harvest: {},
            build: {},
        }

        const techContainer = document.getElementById("tech-container");
        const untiContainer = document.getElementById("unit-container");
        const harvestContainer = document.getElementById("harvest-container");
        const buildContainer = document.getElementById("build-container");

        console.log(MOVES);
        
        for(const move of MOVES) {
            if(move._moveName.includes('Move') 
                || move._moveName.includes('Attack')
                || move._moveName.includes('Capture')
            ) {
                const [, fromIndex] = move._targetName.split(':')[0].split(',');
                if(!map.units[fromIndex]) map.units[fromIndex] = [];
                map.units[fromIndex].push(move);
            }
            else if(move._moveName.includes('Research')) {
                map.tech.push(move);
            }
            else if(move._moveName.includes('Harvest')) {
                const [fromIndex, ] = move._targetName.split(':')[0].split(',');
                if(!map.harvest[fromIndex]) map.harvest[fromIndex] = [];
                map.harvest[fromIndex].push(move);
            }
            else if(move._moveName.includes('Build')) {
                const [fromIndex, ] = move._targetName.split(':')[0].split(',');
                if(!map.build[fromIndex]) map.build[fromIndex] = [];
                map.build[fromIndex].push(move);
            }
        }
       
        const displayMoves = (container, moves, title) => {
            if(!Object.keys(moves).length || moves.length == 0) {
                container.innerHTML = ``;
                return;
            }
            container.innerHTML = `<p>${title}</p>`;
    
            for(const from in moves) {
                const movesAtFrom = moves[from];
                const builddisplay = document.createElement('div');
    
                const [name, index] = movesAtFrom[0]._targetName.split(':')[0].split(',');
                
                builddisplay.innerHTML += `
                <div>
                    <p>${name}</p>
                    ${movesAtFrom.reduce((acc, cur) => {
                        const [name] = cur._moveName.split(':');
                        const [tileIndex, extra] = cur._moveName.split(':')[1].split(',');
                        const targetRuins = Object.values(state.structures).find(x => x.id == 2 && x.tileIndex == Number(tileIndex));
                        // const targetStarfish = Object.values(GAMESTATE.resources).find(x => x.id == 8 && x.tileIndex == Number(tileIndex));
                        const targetUnit = Object.values(state.tribes).find(x => x._units.find(x => x._tileIndex == Number(tileIndex)))?._units.find(x => x._tileIndex == Number(tileIndex));
                        const targetCity = Object.values(state.tribes).find(x => x._cities.find(x => x.tileIndex == Number(tileIndex)))?._cities.find(x => x.tileIndex == Number(tileIndex));
                        return [...acc, `<span>${name} ${
                            cur._moveName.includes('Capture')? targetCity? `city ${targetCity.name}` : 
                                targetRuins? `ruins` : `village` :
                            targetUnit? ` nearby <strong>${ClassNameToId[targetUnit._unitType]}</strong>` :
                            extra? `to (${tileIndex}, ${extra})` : 
                            `at ${tileIndex}`}</span>`];
                    }, []).join('<br>')}
                </div>`
                container.appendChild(builddisplay);
            }
        }

        if(map.tech.length) {
            techContainer.innerHTML = `<p>Technology</p>`;
            techContainer.innerHTML += map.tech.reduce((acc, cur) => {
                const [name, cost] = cur._moveName.split(':');
                return [...acc, `<span>Research <strong>${cur._targetName}</strong> for ${cost} stars</span>`];
            }, []).join('<br>');
        }
        else {
            techContainer.innerHTML = '';
        }

        displayMoves(harvestContainer, map.harvest, 'Harvest');
        displayMoves(untiContainer, map.units, 'Units');
        displayMoves(buildContainer, map.build, 'Construct');
    });
});