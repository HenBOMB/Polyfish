class MapRenderer {
    constructor(container) {
        this.container = container;
        this.elements = new Map(); // idx -> { ground, unit, city, structures, resources, fog }
        this.selectedIdx = null;

        // Territory SVG Layer
        let svg = this.container.querySelector('.territory-overlay');
        if (!svg) {
            svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            svg.classList.add('territory-overlay');
            svg.style.position = 'absolute';
            svg.style.top = '0';
            svg.style.left = '0';
            // Large size to cover map scrolling, or setup overflow: visible
            svg.style.width = '1px';
            svg.style.height = '1px';
            svg.style.overflow = 'visible';
            svg.style.pointerEvents = 'none';
            svg.style.zIndex = '500'; // Below units (5000) but above ground? Ground is 0..fog. 
            // Ground zIndex is pos.y. We need borders to be visible. pos.y is large.
            // Actually zIndex in CSS works differently. If tiles are position:absolute, they interleave based on z-index.
            // SVG is one big element. If we put z-index 500, it might be behind some tiles and in front of others?
            // Problem: SVG is a single flat layer. Tiles are sorted by Z.
            // If we put SVG at z-index 500, it cuts through the map 3D-wise?
            // Solution: Borders should probably be drawn PER TILE as an overlay div/img, or SVG needs to be very high Z to sit on top of everything?
            // "borders of the city... around the edges". Usually this is on the ground.
            // If we draw on top of everything (z 10000), it might overlay units?
            // Dash border usually implies ground level.
            // Let's try z-index 20 (just above ground which is 0-10, but below objects like cities/units which are 4000+).
            // Actually objects have zIndex = pos.y + offsets.
            // pos.y can be 0 to ~2000+.
            // If SVG is z-index 20, it will be behind "southern" tiles.
            // Isometric map layering is tricky with a single SVG.
            // ALTERNATIVE: Create individual border divs for each tile edge?
            // OR: SVG per tile?
            // User asked for "dashed outline".
            // Let's stick to SVG on top (z-index 6000) for now. It will overlay units slightly if they stand on the edge, but that's acceptable for a prompt.
            svg.style.zIndex = '1500'; // Above ground, below most units/cities/trees ideally? 
            // Trees are 2000+. Cities 4000. Units 5000.
            // So 1500 is good: above ground, below objects.
            this.container.appendChild(svg);
        }
        this.territorySvg = svg;
    }

    clear() {
        this.container.innerHTML = '';
        this.elements.clear();
    }

    getPos(x, y) {
        const posX = (x - y) * (tileSize / 2 - TILE_OFFSET);
        const posY = (x + y) * (tileSize / 4 + TILE_OFFSET);
        return { x: posX, y: posY };
    }

    render(state, legalMoves, povOverride = null) {
        const currentTribeId = povOverride !== null ? povOverride : state.settings.currentPlayerTurnId;
        this.currentPovId = currentTribeId;

        const unitsByIndex = {};
        const citiesByIndex = {};
        const unitsByCityId = {};
        Object.values(state.tribes).forEach(tribe => {
            (tribe.units || []).forEach(u => {
                unitsByIndex[u.coords.idx] = { ...u, tribe };
                const cityId = u.cityId ?? u.homeCoords?.idx;
                if (cityId !== undefined && cityId !== -1) {
                    if (!unitsByCityId[cityId]) unitsByCityId[cityId] = 0;
                    unitsByCityId[cityId]++;
                }
            });
            (tribe.cities || []).forEach(c => citiesByIndex[c.tileIndex] = { ...c, tribe });
        });

        const allTiles = Object.values(state.map.tiles).sort((a, b) => a.coords.idx - b.coords.idx);

        allTiles.forEach(tile => {
            this.renderTile(tile, state, unitsByIndex, citiesByIndex, currentTribeId, unitsByCityId);
        });

        this.renderMoveOverlays(legalMoves);
        this.renderCaptureIndicators(legalMoves);
        this.renderTerritoryBorders(state);
    }

    renderTerritoryBorders(state) {
        // Clear existing
        this.territorySvg.innerHTML = '';

        const tiles = Object.values(state.map.tiles);
        const tileMap = {};
        tiles.forEach(t => tileMap[`${t.coords.x},${t.coords.y}`] = t);

        const getOwner = (x, y) => {
            const t = tileMap[`${x},${y}`];
            return t ? (t.owner > 0 ? t.owner : 0) : 0;
        };

        tiles.forEach(tile => {
            if (tile.owner <= 0) return; // Neutral land has no borders

            // FOW Check
            const isExplored = !ENABLE_FOW || (tile.explorers && tile.explorers.includes(this.currentPovId));
            if (!isExplored) return;

            const { x, y } = tile.coords;
            const owner = tile.owner;
            const pos = this.getPos(x, y);

            // Neighbors:
            // N: (x, y-1) -> NE Edge
            // E: (x+1, y) -> SE Edge
            // S: (x, y+1) -> SW Edge
            // W: (x-1, y) -> NW Edge

            const neighbors = [
                { nx: x, ny: y - 1, edge: 'NE' },
                { nx: x + 1, ny: y, edge: 'SE' },
                { nx: x, ny: y + 1, edge: 'SW' },
                { nx: x - 1, ny: y, edge: 'NW' }
            ];

            const color = TRIBE_COLORS[state.tribes[owner]?.type] || '#ffffff';

            neighbors.forEach(({ nx, ny, edge }) => {
                const nOwner = getOwner(nx, ny);
                if (nOwner !== owner) {
                    // Draw Border
                    let x1, y1, x2, y2;
                    // Diamond offsets relative to pos:
                    // Top: 64, 0
                    // Right: 128, 32
                    // Bottom: 64, 64
                    // Left: 0, 32

                    if (edge === 'NE') { // Top -> Right
                        x1 = 64; y1 = -4; x2 = 124; y2 = 32;
                    } else if (edge === 'SE') { // Right -> Bottom
                        x1 = 124; y1 = 32; x2 = 64; y2 = 68;
                    } else if (edge === 'SW') { // Bottom -> Left
                        x1 = 64; y1 = 68; x2 = 4; y2 = 32;
                    } else if (edge === 'NW') { // Left -> Top
                        x1 = 4; y1 = 32; x2 = 64; y2 = -4;
                    }

                    // Add absolute position
                    x1 += pos.x; y1 += pos.y;
                    x2 += pos.x; y2 += pos.y;

                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', x1);
                    line.setAttribute('y1', y1);
                    line.setAttribute('x2', x2);
                    line.setAttribute('y2', y2);
                    line.setAttribute('stroke', color);
                    line.setAttribute('stroke-width', '4'); // Nice thick border
                    line.setAttribute('stroke-dasharray', '8 4'); // Dashed
                    line.setAttribute('stroke-linecap', 'round');
                    line.setAttribute('opacity', '0.8');
                    this.territorySvg.appendChild(line);
                }
            });
        });
    }

    renderTile(tile, state, unitsByIndex, citiesByIndex, currentTribeId, unitsByCityId) {
        const idx = tile.coords.idx;
        const { x, y } = tile.coords;
        const pos = this.getPos(x, y);

        let data = this.elements.get(idx);
        if (!data) {
            data = { layers: {} };
            this.elements.set(idx, data);
        }

        // FOW Logic
        let isExplored = true;
        if (ENABLE_FOW) {
            if (tile.explorers && !tile.explorers.includes(currentTribeId)) isExplored = false;
        }

        if (!isExplored) {
            this.updateLayer(idx, 'ground', 'terrain/tiles/undiscovered', pos, 10, ['ground', 'undiscovered']);
            this.removeLayer(idx, 'unit');
            this.removeLayer(idx, 'city');
            this.removeLayer(idx, 'city-stats');
            this.removeLayer(idx, 'resource');
            this.removeLayer(idx, 'structure');
            this.removeLayer(idx, 'ambient');
            return;
        }

        // Ground
        const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, null, null, 'terrain/tiles/ice'][tile.type]
            || `terrain/tiles/ground_${tile.climate}`;

        const groundEl = this.updateLayer(idx, 'ground', tilefile, pos, 0, ['ground']);
        groundEl.dataset.tileIdx = idx;

        groundEl.classList.remove('fog');

        // Ambient (Mountains/Forests)
        if (tile.type === 4) { // Mountain
            this.updateLayer(idx, 'ambient', `terrain/mountains/mountain_${tile.climate}`, pos, 1200, ['mountain']);
        } else if (tile.type === 5) { // Forest
            this.updateLayer(idx, 'ambient', `terrain/forests/Forest_${tile.climate}`, pos, 2000, ['forest']);
        } else {
            this.removeLayer(idx, 'ambient');
        }

        // Resources & Structures
        const struct = state.structures[idx];
        const res = state.resources[idx];

        if (struct && struct.type === 71) { // Road
            this.updateLayer(idx, 'road', 'misc/Road', pos, 1500, ['structure', 'road']);
        } else {
            this.removeLayer(idx, 'road');
        }

        const povTribe = state.tribes[currentTribeId];
        if (res && isResourceVisible(res.type, povTribe)) {
            const resFile = getResourceFile(res.type, tile.climate);
            const resClass = { 1: 'animal', 2: 'crop', 3: 'fish', 5: 'metal', 6: 'fruit' }[res.type];
            const resClasses = ['resource', resClass];

            // Highlight if harvestable
            if (currentLegalMoves.some(m => m.moveType === 5 && (m.target === idx || m.target_index === idx))) {
                resClasses.push('interactive-outline');
            }

            this.updateLayer(idx, 'resource', resFile, pos, 2500, resClasses);
        } else {
            this.removeLayer(idx, 'resource');
        }

        if (struct && struct.type !== 71) {
            // Skip rendering village if there's a city here
            if (struct.type === 1 && citiesByIndex[idx]) {
                this.removeLayer(idx, 'structure');
            } else {
                const structFile = getStructureFile(struct.type, tile.climate);
                const classes = ['structure'];
                if (struct.type === 1) classes.push('village');
                if (struct.type === 2) classes.push('ruins');
                if (struct.type === 29) classes.push('monument');
                this.updateLayer(idx, 'structure', structFile, pos, 2000, classes);
            }
        } else {
            this.removeLayer(idx, 'structure');
        }

        // Cities
        const city = citiesByIndex[idx];
        if (city) {
            const tribeName = TRIBE_ID_2_NAME[city.tribe.type];
            const climateIndex = CLIMATE_IDS.indexOf(tribeName);
            const tribeColor = TRIBE_COLORS[city.tribe.type] || 'rgba(0,0,0,0.5)';

            const cityEl = this.updateLayer(idx, 'city', `buildings/${tribeName}/Default/Houses/House_${climateIndex}_5`, pos, 4000, ['city']);
            const statsEl = this.updateLayer(idx, 'city-stats', '', pos, 4100, ['city-stats-layer']);
            statsEl.style.backgroundImage = 'none'; // No background for stats container

            // Calculate segments (Polytopia style: dots for units, fill for population)
            const totalSegments = city.level + 1;
            const filledSegments = city.progress;
            const unitCount = unitsByCityId[idx] || 0;
            let segmentsHtml = '';
            for (let i = 0; i < totalSegments; i++) {
                const isFilled = i < filledSegments;
                const hasUnit = i < unitCount;
                segmentsHtml += `<div class="growth-segment ${isFilled ? 'filled' : 'empty'} ${hasUnit ? 'has-unit' : ''}"></div>`;
            }

            const income = getCityProduction(state, city);
            const isCapital = tile.capitalOf > 0;
            const isConnected = city.connectedToCapital;

            statsEl.innerHTML = `
                <div class="city-container">
                    <div class="city-banner" style="background-color: ${tribeColor}66; --tribe-outline: ${tribeColor}">
                        <div class="banner-left">
                            ${isCapital ? '<span class="crown">👑</span>' : ''}
                            <span class="name">${city.name?.replace(tribeName, '').trim() || 'City'}</span>
                        </div>
                        <div class="income-badge">
                            <span class="star">⭐</span>
                            <span class="val">${income}</span>
                        </div>
                    </div>
                    <div class="city-growth-bar">
                        ${segmentsHtml}
                    </div>
                </div>
            `;
        } else {
            this.removeLayer(idx, 'city');
            this.removeLayer(idx, 'city-stats');
        }

        // Units
        const unit = (isExplored || !ENABLE_FOW) ? unitsByIndex[idx] : null;
        if (unit) {
            const tribeName = TRIBE_ID_2_NAME[unit.tribe.type];
            const className = ClassNameToId[unit.unitType || unit.type];
            if (className) {
                const classes = ['unit'];
                if (unit.moved || unit.attacked) classes.push('exausted');
                if (unit.flipped) classes.push('flipped');
                if (this.selectedIdx === idx) classes.push('selected-unit-highlight');

                const unitEl = this.updateLayer(idx, 'unit', `units/${tribeName}/default/${tribeName}_default_${className}`, pos, 5000, classes);
                unitEl.innerHTML = `<div class="health">${Math.floor(unit.health / 10)}</div>`;
            }
        } else {
            this.removeLayer(idx, 'unit');
        }
    }

    updateLayer(idx, layerName, filename, pos, zIndex, classes = []) {
        let data = this.elements.get(idx);
        let el = data.layers[layerName];

        if (!el) {
            el = document.createElement('div');
            el.classList.add('tile');
            this.container.appendChild(el);
            data.layers[layerName] = el;
        }

        el.style.backgroundImage = `url('textures/${filename}.png')`;
        el.style.left = `${pos.x}px`;
        el.style.top = `${pos.y}px`;
        el.style.zIndex = Math.floor(pos.y + zIndex);

        // Reset classes and apply new ones
        el.className = 'tile';
        classes.forEach(c => { if (c) el.classList.add(c) });

        // If this is the ground layer, ensure it has a click mask
        if (layerName === 'ground') {
            this.updateClickMask(idx, pos);
        }

        return el;
    }

    updateClickMask(idx, pos) {
        let data = this.elements.get(idx);
        let mask = data.layers['click-mask'];

        if (!mask) {
            mask = document.createElement('div');
            mask.classList.add('tile', 'click-mask');
            this.container.appendChild(mask);
            data.layers['click-mask'] = mask;
        }

        mask.style.left = `${pos.x}px`;
        mask.style.top = `${pos.y}px`;
        mask.style.zIndex = Math.floor(pos.y + 10000); // Topmost for interaction

        mask.onclick = (e) => {
            const unit = this.getUnitAt(idx);
            this.handleTileClick(e, idx, unit, this.isTileVisible(idx));
        };
        this.setupHover(mask, idx, this.isTileVisible(idx));
    }

    isTileVisible(idx) {
        if (!ENABLE_FOW) return true;
        const currentPov = (this.currentPovId !== undefined && this.currentPovId !== null) ? this.currentPovId : (GAME_STATE.settings?.currentPlayerTurnId ?? 1);
        return GAME_STATE.tiles[idx].explorers && GAME_STATE.tiles[idx].explorers.includes(currentPov);
    }

    removeLayer(idx, layerName) {
        const data = this.elements.get(idx);
        if (data && data.layers[layerName]) {
            data.layers[layerName].remove();
            delete data.layers[layerName];
        }
    }

    handleTileClick(e, idx, unit, isExplored) {
        e.stopPropagation();

        if (ENABLE_FOW) {
            const tile = GAME_STATE.tiles[idx];
            const currentTribeId = this.currentPovId;
            const isExplored = tile && tile.explorers && tile.explorers.includes(currentTribeId);

            if (!isExplored) {
                this.selectedIdx = null;
                selectedUnitIdx = null;
                this.render(GAME_STATE, currentLegalMoves);
                updateSelectionInfo();
                return;
            }
        }

        const hasUnit = !!unit;
        const struct = GAME_STATE.structures[idx];
        const resource = GAME_STATE.resources[idx];
        // Village (3) or City (4)
        const isCity = struct && (struct.structureType === 3 || struct.structureType === 4 || struct.type === 3 || struct.type === 4);
        const hasActions = isCity || !!resource;

        // Toggle selection with cycle: Unit -> City -> None
        if (this.selectedIdx !== idx) {
            // New tile selected
            this.selectedIdx = idx;
            selectedUnitIdx = hasUnit ? idx : null;
        } else {
            // Same tile clicked again
            if (selectedUnitIdx === idx) {
                // Check if we can trigger a self-action/ability (e.g. Recover)
                // Filter for Ability moves (Type 3)
                // Exclude Disband (Type 8) to avoid accidents
                // Prioritize: Promote > Recover > Others
                const abilityMoves = currentLegalMoves.filter(m =>
                    m.src === idx &&
                    m.moveType === 3 &&
                    m.type !== 8 // Exclude Disband
                );

                if (abilityMoves.length > 0) {
                    // Sort by priority (Recover=7, Promote=16, HealOthers=9)
                    // We want Promote first, then Recover, then others.
                    abilityMoves.sort((a, b) => {
                        const score = (type) => {
                            if (type === 16) return 100; // Promote
                            if (type === 7) return 50;  // Recover
                            if (type === 9) return 40;  // HealOthers
                            return 0; // Others
                        };
                        return score(b.type) - score(a.type);
                    });

                    // Execute the best move
                    playMove(abilityMoves[0]);
                    return;
                }

                // Was on Unit, move to City if it has actions
                if (hasActions) {
                    selectedUnitIdx = null;
                } else {
                    // No city, deselect
                    this.selectedIdx = null;
                    selectedUnitIdx = null;
                }
            } else {
                // Was on City (or tile with neither), deselect
                this.selectedIdx = null;
                selectedUnitIdx = null;
            }
        }

        this.render(GAME_STATE, currentLegalMoves);
        updateSelectionInfo(e.clientX, e.clientY);
    }

    setupHover(el, idx, isExplored) {
        el.onmouseenter = (e) => {
            if (!this.isTileVisible(idx)) return;
            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;

            hoverEl.classList.remove('hidden');
            const unit = (isExplored || !ENABLE_FOW) ? this.getUnitAt(idx) : null;
            const struct = GAME_STATE.structures[idx];
            const resource = GAME_STATE.resources[idx];
            const isCity = tile.owner > 0 && struct && StructureTypes[struct.type] === 'Village';
            const tribe = TRIBE_ID_2_NAME[GAME_STATE.tribes[tile.owner]?.type];

            let html = `<strong>Tile ${idx} (${tile.coords.x}, ${tile.coords.y})</strong><br>`;
            html += `⛰️ ${CLIMATE_IDS[tile.climate]} ${TerrainType[tile.type] || tile.type}<br>`;
            if (unit) html += `🪖 ${TRIBE_ID_2_NAME[unit.tribe.type]} ${ClassNameToId[unit.type || unit.unitType]} (${Math.floor(unit.health / 10)}/${Math.floor(unit.maxHealth / 10)})<br>`;
            if (struct) html += `🗼 ${tile.capitalOf > 0 ? `${tribe} Capital` : isCity ? 'City' : StructureTypes[struct.type] || struct.type}<br>`;
            const povTribe = GAME_STATE.tribes[this.currentPovId];
            if (resource && isResourceVisible(resource.type, povTribe)) {
                html += `${ResourceEmojis[resource.type] || '🥝'} ${ResourceTypes[resource.type] || resource.type}<br>`;
            }

            hoverEl.innerHTML = html;

            // Highlight the visual tile and its contents
            const data = this.elements.get(idx);
            if (data) {
                // User Request: Don't highlight exhausted friendly units
                let skipHighlight = false;
                const activeTribeId = GAME_STATE.settings.currentPlayerTurnId;
                const activeTribe = GAME_STATE.tribes[activeTribeId];

                // Check if it's a friendly unit
                if (unit && activeTribe && unit.tribe.type === activeTribe.type) {
                    // Check if active (has moves)
                    const hasMoves = currentLegalMoves.some(m => m.src === idx || m.src_index === idx);
                    if (!hasMoves) skipHighlight = true;
                }

                if (!skipHighlight) {
                    if (data.layers['ground']) {
                        data.layers['ground'].classList.add('tile-hover-highlight');
                    }
                    // Apply shader outline to sprites
                    ['unit', 'city', 'structure', 'resource'].forEach(layer => {
                        if (data.layers[layer]) data.layers[layer].classList.add('hover-outline');
                    });
                }
            }
        };

        el.onmousemove = (e) => {
            hoverEl.style.left = `${e.clientX + 15}px`;
            hoverEl.style.top = `${e.clientY + 15}px`;
        };

        el.onmouseleave = () => {
            hoverEl.classList.add('hidden');
            const data = this.elements.get(idx);
            if (data) {
                if (data.layers['ground']) {
                    data.layers['ground'].classList.remove('tile-hover-highlight');
                }
                ['unit', 'city', 'structure', 'resource'].forEach(layer => {
                    if (data.layers[layer]) data.layers[layer].classList.remove('hover-outline');
                });
            }
        };
    }

    getUnitAt(idx) {
        let found = null;
        Object.values(GAME_STATE.tribes).forEach(tribe => {
            const u = (tribe.units || []).find(u => u.coords.idx === idx);
            if (u) found = { ...u, tribe };
        });
        return found;
    }

    renderMoveOverlays(legalMoves) {
        this.clearMoveHighlight();
        document.querySelectorAll('.move-overlay').forEach(el => el.remove());
        document.querySelectorAll('.combat-preview:not(.temp-highlight)').forEach(el => el.remove());
        if (this.selectedIdx === null) return;

        const unitMoves = legalMoves.filter(m => m && typeof m === 'object' && m.src === this.selectedIdx);

        unitMoves.forEach(move => {
            const targetIdx = move.target;
            if (targetIdx === undefined || targetIdx === null) return;

            const targetTile = GAME_STATE.tiles[targetIdx];
            if (!targetTile) return;

            const pos = this.getPos(targetTile.coords.x, targetTile.coords.y);
            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'move-overlay');
            overlay.style.left = `${pos.x}px`;
            overlay.style.top = `${pos.y}px`;
            overlay.style.zIndex = Math.floor(pos.y + 20000);

            const img = document.createElement('img');
            const isAttack = move.moveType === 2;
            img.src = isAttack ? 'textures/misc/attackTarget.png' : 'textures/misc/moveTarget.png';
            img.style.width = '128px';
            overlay.appendChild(img);

            // Add combat preview for attack moves when predictions are enabled (permanent)
            if (isAttack && SHOW_PREDICTIONS) {
                this.renderCombatPreview(this.selectedIdx, targetIdx, pos);
            }

            overlay.onclick = (e) => {
                e.stopPropagation();
                playMove(move);
            };

            overlay.onmouseenter = () => this.highlightMove(move);
            overlay.onmouseleave = () => this.clearMoveHighlight();

            this.container.appendChild(overlay);
        });
    }

    renderCaptureIndicators(legalMoves) {
        // Clear existing capture indicators
        document.querySelectorAll('.capture-indicator').forEach(el => el.remove());

        if (!legalMoves) return;
        const captureMoves = legalMoves.filter(m => m.moveType === 8);

        captureMoves.forEach(move => {
            const idx = move.src; // Capture happens at unit's location
            if (idx === undefined || idx === null) return;

            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;

            const pos = this.getPos(tile.coords.x, tile.coords.y);

            const indicator = document.createElement('div');
            indicator.className = 'tile capture-indicator';
            indicator.style.left = `${pos.x}px`;
            indicator.style.top = `${pos.y}px`;
            indicator.style.zIndex = Math.floor(pos.y + 25000); // Very high z-index
            indicator.title = "Click to Capture!";

            const img = document.createElement('img');
            img.src = 'textures/misc/Capture.png';
            img.style.width = '64px'; // Smaller than full tile
            img.style.height = '64px';
            img.style.position = 'absolute';
            img.style.left = '32px';
            img.style.top = '32px';
            img.className = 'pulse-animation'; // Add CSS animation class

            indicator.appendChild(img);

            indicator.onclick = (e) => {
                e.stopPropagation();
                playMove(move);
            };

            // Add to container
            this.container.appendChild(indicator);
        });
    }

    async renderCombatPreview(srcIdx, targetIdx, pos, isTemp = false) {
        try {
            const res = await fetch('/simulate/attack', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ src: srcIdx, target: targetIdx })
            });
            const preview = await res.json();
            if (preview.error) return;

            const previewEl = document.createElement('div');
            previewEl.classList.add('combat-preview');
            if (isTemp) previewEl.classList.add('temp-highlight');

            // Color based on outcome
            if (preview.defenderDies && !preview.attackerDies) {
                previewEl.classList.add('combat-favorable');
            } else if (preview.attackerDies) {
                previewEl.classList.add('combat-deadly');
            } else {
                previewEl.classList.add('combat-risky');
            }

            // Show damage numbers (Backend sends them in 10x scale, e.g. 45 for 4.5 hearts)
            const dmgDef = (preview.damageToDefender / 10).toFixed(1);
            const dmgAtk = (preview.damageToAttacker / 10).toFixed(1);

            previewEl.innerHTML = `⚔️ ${dmgDef} / -${dmgAtk}`;
            previewEl.style.left = `${pos.x}px`;
            previewEl.style.top = `${pos.y}px`;
            previewEl.style.zIndex = Math.floor(pos.y + 25000);
            this.container.appendChild(previewEl);
        } catch (e) {
            console.error("Combat simulation failed", e);
        }
    }

    highlightMove(move) {
        this.clearMoveHighlight();
        if (!move) return;

        // Unified move type detection
        const moveType = move.moveType !== undefined ? move.moveType :
            (move.type !== undefined ? 7 :
                (move.type !== undefined ? 6 :
                    (move.type !== undefined ? 3 :
                        (move.type !== undefined ? 9 : 0))));

        const srcIdx = move.src;
        const targetIdx = move.target !== undefined ? move.target : move.src;

        // 1. Highlight source tile
        if (srcIdx !== undefined && srcIdx !== null) {
            const srcData = this.elements.get(srcIdx);
            if (srcData && srcData.layers['ground']) srcData.layers['ground'].classList.add('source-tile-highlight');
        }

        // 2. Highlight target tile and show ghost/preview
        if (targetIdx !== undefined && targetIdx !== null) {
            const targetTile = GAME_STATE.tiles[targetIdx];
            if (targetTile) {
                const pos = this.getPos(targetTile.coords.x, targetTile.coords.y);

                // Show Path Highlight
                // REMOVED per user request
                const overlay = document.createElement('div');
                overlay.classList.add('tile', 'move-path-highlight', 'temp-highlight');
                overlay.style.left = `${pos.x}px`;
                overlay.style.top = `${pos.y}px`;
                overlay.style.zIndex = Math.floor(pos.y + 100);
                this.container.appendChild(overlay);

                // Show Ghost Unit (only for movement/transport)
                if (moveType === 1 || moveType === 2) {
                    const srcUnit = (srcIdx !== undefined) ? this.getUnitAt(srcIdx) : null;
                    if (srcUnit) {
                        const tribeName = TRIBE_ID_2_NAME[srcUnit.tribe.type];
                        const className = ClassNameToId[srcUnit.unitType || srcUnit.type];
                        if (tribeName && className) {
                            const classes = ['unit', 'ghost-unit', 'temp-highlight'];
                            if (srcUnit.flipped) classes.push('flipped');
                            this.updateLayer(targetIdx, 'ghost-unit', `units/${tribeName}/default/${tribeName}_default_${className}`, pos, 5500, classes);
                        }

                        // Show temporary combat preview on hover (always use backend simulator now)
                        if (moveType === 2) {
                            this.renderCombatPreview(srcIdx, targetIdx, pos, true);
                        }
                    }
                }
            }
        }
    }

    clearMoveHighlight() {
        document.querySelectorAll('.temp-highlight').forEach(el => el.remove());
        document.querySelectorAll('.source-tile-highlight').forEach(el => el.classList.remove('source-tile-highlight'));
        this.elements.forEach(data => {
            if (data.layers['ghost-unit']) {
                data.layers['ghost-unit'].remove();
                delete data.layers['ghost-unit'];
            }
        });
    }

    renderMCTSHeatmap(mctsAnalysis) {
        // Clear existing heatmap overlays
        document.querySelectorAll('.mcts-overlay').forEach(el => el.remove());

        if (!mctsAnalysis || !mctsAnalysis.evaluations || mctsAnalysis.evaluations.length === 0) return;
        if (!SHOW_PREDICTIONS) return;

        const evaluations = mctsAnalysis.evaluations;
        const maxVisits = Math.max(...evaluations.map(e => e.visits));
        const gameSize = GAME_STATE.settings?.size || 11;

        // Group evaluations by target tile for aggregation
        const byTarget = {};
        evaluations.forEach(ev => {
            const key = ev.target >= 0 ? ev.target : ev.src;
            if (!byTarget[key]) {
                byTarget[key] = { totalVisits: 0, weightedWinRate: 0, count: 0 };
            }
            byTarget[key].totalVisits += ev.visits;
            byTarget[key].weightedWinRate += ev.win_rate * ev.visits;
            byTarget[key].count++;
        });

        Object.entries(byTarget).forEach(([tileIdx, data]) => {
            const idx = parseInt(tileIdx);
            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;

            const visits = data.totalVisits;
            const winRate = data.weightedWinRate / visits;

            // Normalize opacity by visits (0.3 to 0.9)
            const opacity = 0.3 + (visits / maxVisits) * 0.6;

            // Determine color class based on win rate
            // win_rate is from evaluator which can be large positive/negative
            // Normalize to 0-1 range approximately
            let colorClass = 'neutral';
            if (winRate > 50) colorClass = 'good';
            else if (winRate < -50) colorClass = 'risky';

            const pos = this.getPos(tile.coords.x, tile.coords.y);
            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'mcts-overlay', colorClass);
            overlay.style.setProperty('--opacity', opacity.toFixed(2));
            overlay.style.left = `${pos.x}px`;
            overlay.style.top = `${pos.y}px`;
            overlay.style.zIndex = Math.floor(pos.y + 15000);

            // Add label with visit count
            const label = document.createElement('div');
            label.classList.add('mcts-label');
            const displayRate = winRate > 0 ? `+${winRate.toFixed(0)}` : winRate.toFixed(0);
            label.innerHTML = `<span class="mcts-visits">${Math.round(visits)}</span>`;
            overlay.appendChild(label);

            this.container.appendChild(overlay);
        });
    }

    renderFOWPredictions(prediction) {
        // Clear existing prediction overlays
        document.querySelectorAll('.village-marker').forEach(el => el.remove());
        document.querySelectorAll('.predicted-terrain').forEach(el => el.remove());

        if (!prediction || !SHOW_PREDICTIONS) return;
        console.log("Rendering Predictions:", prediction);

        // Render predicted villages
        const villages = prediction._villages || prediction.villages;
        if (villages) {
            Object.entries(villages).forEach(([tileIdx, [tribeType, isVillage]]) => {
                const idx = parseInt(tileIdx);
                const tile = GAME_STATE.tiles[idx];
                if (!tile) return;

                const pos = this.getPos(tile.coords.x, tile.coords.y);
                const marker = document.createElement('div');
                marker.classList.add('village-marker');
                marker.textContent = '🏕️';
                marker.title = `Predicted village (${TRIBE_ID_2_NAME[tribeType] || 'Unknown'})`;
                marker.style.left = `${pos.x}px`;
                marker.style.top = `${pos.y}px`;
                marker.style.zIndex = Math.floor(pos.y + 16000);
                this.container.appendChild(marker);
            });
        }

        // Render predicted terrain
        const terrain = prediction._terrain || prediction.terrain;
        if (terrain) {
            const currentTribeId = this.currentPovId || 1;
            Object.entries(terrain).forEach(([tileIdx, [terrainType, climate]]) => {
                const idx = parseInt(tileIdx);
                const tile = GAME_STATE.tiles[idx];
                if (!tile) return;

                // Only show if NOT explored (fog/clouds)
                const isExplored = tile.explorers && tile.explorers.includes(currentTribeId);
                if (isExplored) return;

                const pos = this.getPos(tile.coords.x, tile.coords.y);
                const overlay = document.createElement('div');
                overlay.classList.add('tile', 'predicted-terrain');

                // Map terrain type to texture
                // 1: Water, 2: Ocean, 3: Field, 4: Mountain, 5: Forest, 6: Ice
                const tilefile = [null, 'terrain/water/water', 'terrain/water/ocean', null, 'terrain/mountains/mountain', 'terrain/forests/Forest', 'terrain/tiles/ice'][terrainType]
                    || `terrain/tiles/ground_${climate}`;

                // Handle specific suffixes for mountain/forest
                let finalFile = tilefile;
                if (terrainType === 4) finalFile = `terrain/mountains/mountain_${climate}`;
                if (terrainType === 5) finalFile = `terrain/forests/Forest_${climate}`;

                overlay.style.backgroundImage = `url('textures/${finalFile}.png')`;
                overlay.style.left = `${pos.x}px`;
                overlay.style.top = `${pos.y}px`;
                overlay.style.zIndex = Math.floor(pos.y + 20); // Just above ground

                this.container.appendChild(overlay);
            });
        }

        // Render suspected enemy capitals
        const suspects = prediction._enemy_capital_suspects || prediction.enemyCapitalSuspects || prediction._enemyCapitalSuspects;
        if (suspects && suspects.length > 0) {
            suspects.forEach(idx => {
                const tile = GAME_STATE.tiles[idx];
                if (!tile) return;

                const pos = this.getPos(tile.coords.x, tile.coords.y);
                const marker = document.createElement('div');
                marker.classList.add('village-marker');
                marker.textContent = '👑';
                marker.title = 'Suspected enemy capital';
                marker.style.left = `${pos.x}px`;
                marker.style.top = `${pos.y}px`;
                marker.style.zIndex = Math.floor(pos.y + 16000);
                this.container.appendChild(marker);
            });
        }
    }

    renderExplorerPath(path, revealed) {
        // Clear previous explorer paths
        document.querySelectorAll('.explorer-path-overlay').forEach(el => el.remove());

        if (!path || path.length === 0) return;

        // Render Path
        for (let i = 0; i < path.length - 1; i++) {
            const currentIdx = path[i];
            const nextIdx = path[i + 1];

            const t1 = GAME_STATE.tiles[currentIdx];
            const t2 = GAME_STATE.tiles[nextIdx];
            if (!t1 || !t2) continue;

            const p1 = this.getPos(t1.coords.x, t1.coords.y);
            // const p2 = this.getPos(t2.coords.x, t2.coords.y);

            // Draw an arrow or marker at the next tile indicating flow
            // Use a simple dot/arrow overlay on the 'next' tile
            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'explorer-path-overlay');
            overlay.style.left = `${(p1.x + 64)}px`; // Centerish? 
            // Wait, coordinate system is isometric. 
            // Let's just highlight the tiles in the path with numbers.

        }

        // Better: Highlight path tiles with step numbers
        path.forEach((idx, step) => {
            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;
            const pos = this.getPos(tile.coords.x, tile.coords.y);

            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'explorer-path-overlay');
            overlay.style.left = `${pos.x}px`;
            overlay.style.top = `${pos.y}px`;
            overlay.style.zIndex = Math.floor(pos.y + 17000);

            // Visual style
            overlay.style.pointerEvents = 'none';
            overlay.innerHTML = `<div style="
                position: absolute; 
                left: 50%; top: 50%; 
                transform: translate(-50%, -50%);
                font-size: 24px; 
                font-weight: bold; 
                color: #00ff00; 
                text-shadow: 0 0 4px black;">
                ${step}
            </div>`;

            this.container.appendChild(overlay);
        });

        // Highlight revealed tiles
        revealed.forEach(idx => {
            // explicit 'revealed-highlight'
            const tile = GAME_STATE.tiles[idx];
            if (!tile) return;
            const pos = this.getPos(tile.coords.x, tile.coords.y);

            const overlay = document.createElement('div');
            overlay.classList.add('tile', 'explorer-path-overlay');
            overlay.style.left = `${pos.x}px`;
            overlay.style.top = `${pos.y}px`;
            overlay.style.zIndex = Math.floor(pos.y + 16500);
            overlay.style.backgroundColor = 'rgba(0, 255, 0, 0.2)';
            overlay.style.border = '2px dashed #00ff00';

            this.container.appendChild(overlay);
        });
    }

    renderAnalysis(analysis) {
        // Clear analysis overlays
        document.querySelectorAll('.analysis-overlay').forEach(el => el.remove());

        if (!analysis || !SHOW_PREDICTIONS) return;

        // Render Expansion Values
        if (analysis.tileValues) {
            Object.entries(analysis.tileValues).forEach(([tileIdx, value]) => {
                const idx = parseInt(tileIdx);
                const tile = GAME_STATE.tiles[idx];
                if (!tile) return;

                const pos = this.getPos(tile.coords.x, tile.coords.y);
                const overlay = document.createElement('div');
                overlay.classList.add('tile', 'analysis-overlay', 'expansion-overlay');

                // Opacity based on value (clamped 0.2 to 0.7)
                const opacity = Math.min(0.7, Math.max(0.2, value / 10.0)).toFixed(2);
                overlay.style.setProperty('--opacity', opacity);
                overlay.style.backgroundColor = `rgba(138, 43, 226, ${opacity})`; // Violet

                overlay.style.left = `${pos.x}px`;
                overlay.style.top = `${pos.y}px`;
                overlay.style.zIndex = Math.floor(pos.y + 14000); // Below units

                // Add value label
                /*
                const label = document.createElement('div');
                label.classList.add('mcts-label');
                label.innerText = value.toFixed(1);
                overlay.appendChild(label);
                */

                this.container.appendChild(overlay);
            });
        }

        // Render Threats
        if (analysis.threats) {
            analysis.threats.forEach(threat => {
                const idx = threat.targetTile;
                const tile = GAME_STATE.tiles[idx];
                if (!tile) return;

                const pos = this.getPos(tile.coords.x, tile.coords.y);
                const overlay = document.createElement('div');
                overlay.classList.add('tile', 'analysis-overlay', 'threat-overlay');

                overlay.style.left = `${pos.x}px`;
                overlay.style.top = `${pos.y}px`;
                overlay.style.zIndex = Math.floor(pos.y + 18000); // Above units

                const marker = document.createElement('div');
                marker.classList.add('village-marker', 'threat-marker');
                marker.textContent = '⚠️';
                marker.style.textShadow = '0 0 5px red';

                if (threat.threatLevel >= 1.0) {
                    marker.textContent = '💀'; // Lethal
                }

                overlay.appendChild(marker);
                this.container.appendChild(overlay);
            });
        }
    }
}
