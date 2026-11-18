const byId = (id) => document.getElementById(id);
const setup = byId('setup');
const board = byId('board');
const viewport = byId('viewport');
const tribes = [['xin-xi', 'Xin-Xi'], ['imperius', 'Imperius'], ['bardur', 'Bardur'], ['oumaji', 'Oumaji']];
for (const [field, initial] of [['tribe1', 'imperius'], ['tribe2', 'bardur']]) {
  for (const [value, name] of tribes) {
    setup.elements[field].add(new Option(name, value, false, value === initial));
  }
}

let match;
let selected = null;
let busy = false;
let zoom = 1;
let learnedAvailable = false;

async function request(path, data) {
  const response = await fetch(path, data === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: typeof data === 'string' ? data : JSON.stringify(data),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function setError(message) {
  const element = byId('error');
  element.textContent = message;
  element.hidden = !message;
  byId('retry-bot').hidden = !(message && match?.botTurn);
}

function updateLearned() {
  const opponent = setup.elements.opponentPreset;
  const learned = opponent.querySelector('[value=learned]');
  learned.disabled = !learnedAvailable || setup.elements.mapSize.value !== '121';
  if (opponent.value === 'learned' && learned.disabled) opponent.value = 'normal';
}

function fitBoard() {
  if (!match?.size) return;
  const { clientWidth, clientHeight } = viewport;
  const available = Math.min(clientWidth, clientHeight) - 24 - 2 * (match.size - 1);
  const cell = Math.max(24, Math.min(66, available / match.size));
  board.style.setProperty('--cell', Math.round(cell * zoom) + 'px');
}

function formatName(value) {
  return value.replace(/([a-z])([A-Z])/g, '$1 $2');
}

function tileLabel(tile) {
  if (!tile.visible) return 'Unexplored';
  const unit = tile.unit && (formatName(tile.unit.name) + ' · ' + tile.unit.health + ' HP');
  const city = tile.city && (tile.city.name + ' · level ' + tile.city.level);
  return [tile.terrain, city, unit, tile.structure && formatName(tile.structure),
    tile.resource && formatName(tile.resource), tile.road && 'Road'].filter(Boolean).join(' · ');
}

function makeButton(label, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.addEventListener('click', onClick);
  return button;
}

function actionButton(action) {
  const button = makeButton(action.label, () => executeAction(action));
  button.disabled = busy;
  return button;
}

function render() {
  if (!match?.tiles) return;
  const tile = match.tiles.find((tile) => tile.index === selected);
  const actions = match.actions.filter((action) => action.kind !== 'EndTurn');
  const tileActions = selected == null ? [] : actions.filter((action) => (action.source ?? action.target) === selected);
  const destinations = new Set(tileActions.map((action) => action.target).filter((idx) => idx != null && idx !== selected));
  const previouslyFocused = document.activeElement?.dataset.index;
  board.style.setProperty('--size', match.size);
  board.replaceChildren(...[...match.tiles].sort((a, b) => a.index - b.index).map((tile) => {
    const label = String(tile.index % match.size + 1) + ', ' + String(Math.floor(tile.index / match.size) + 1);
    const button = makeButton('', () => selectTile(tile.index));
    button.className = 'tile';
    button.dataset.index = tile.index;
    button.dataset.terrain = tile.visible ? tile.terrain : 'Fog';
    const owner = tile.unit?.owner ?? tile.owner;
    if (owner) button.dataset.owner = owner;
    button.toggleAttribute('data-selected', tile.index === selected);
    button.toggleAttribute('data-target', destinations.has(tile.index));
    button.setAttribute('aria-label', label + ' · ' + tileLabel(tile));
    button.setAttribute('aria-pressed', String(tile.index === selected));
    button.tabIndex = tile.index === (selected ?? 0) ? 0 : -1;
    if (tile.unit) {
      const unit = document.createElement('span');
      unit.className = tile.unit.owner === match.pov ? 'unit' : 'unit enemy';
      unit.textContent = tile.unit.name.slice(0, 2).toUpperCase();
      button.append(unit);
    }
    const marker = tile.city ? (tile.capital ? '◆' : '◇') + tile.city.level :
      ({ Village: '○', Ruin: '△' }[tile.structure] ?? (tile.structure || tile.resource || tile.road ? '·' : ''));
    if (marker) {
      const badge = document.createElement('small');
      badge.textContent = marker;
      button.append(badge);
    }
    return button;
  }));
  if (previouslyFocused != null) board.querySelector('[data-index="' + previouslyFocused + '"]')?.focus({ preventScroll: true });

  let status = 'Player ' + match.active + ' · turn ' + match.turn;
  if (match.botTurn) status = 'Opponent is playing · turn ' + match.turn;
  if (match.result.status === 'won') status = 'Player ' + match.result.winner + ' wins';
  if (match.result.status === 'truncated') status = 'Match ended at the turn limit';
  if (match.replay) status = 'Replay · ' + status;
  byId('status').textContent = status;
  byId('match-detail').textContent = [
    'Might', 'Drylands', match.size + ' × ' + match.size, 'Seed ' + match.seed,
    match.stars != null ? match.stars + ' stars' : null,
    match.production != null ? '+' + match.production + ' per turn' : null,
  ].filter(Boolean).join(' · ');
  byId('tile-title').textContent = tile ? 'Tile ' + (tile.index % match.size + 1) + ', ' +
    (Math.floor(tile.index / match.size) + 1) : 'Board actions';
  byId('tile-detail').textContent = tile ? tileLabel(tile) : 'Select a tile to inspect it and see its available actions.';
  byId('tile-actions').replaceChildren(...tileActions.map(actionButton));
  byId('global-actions').replaceChildren(...actions.filter((action) => action.source == null && action.target == null).map(actionButton));
  const endTurn = match.actions.find((action) => action.kind === 'EndTurn');
  byId('end-turn').disabled = busy || !endTurn;
  byId('end-turn').onclick = () => endTurn && executeAction(endTurn);
  byId('save').disabled = busy || !!match.replay;
  byId('replay').hidden = !match.replay;
  if (match.replay) {
    const { position, total } = match.replay;
    byId('previous').disabled = busy || position <= 0;
    byId('next').disabled = busy || position >= total;
    byId('cursor').value = position + ' / ' + total;
  }
  fitBoard();
}

async function transact(operation, finishBotTurn = true) {
  if (busy) return;
  busy = true;
  setError('');
  render();
  try {
    match = await operation();
    render();
    if (finishBotTurn) {
      while (match.botTurn) {
        match = await request('/research/bot', {});
      }
    }
  } catch (error) {
    setError(error.message);
  } finally {
    busy = false;
    render();
  }
}

function selectTile(index) {
  if (!match) return;
  const proposed = selected == null ? [] : match.actions.filter((action) =>
    action.source === selected && action.target === index && selected !== index);
  if (proposed.length === 1 && !busy) {
    executeAction(proposed[0]);
    return;
  }
  selected = index;
  render();
  board.querySelector('[data-index="' + index + '"]')?.focus({ preventScroll: true });
}

function executeAction(action) {
  return transact(() => request('/research/move', { revision: match.revision, action: action.key }));
}

setup.elements.mapSize.addEventListener('change', updateLearned);
setup.addEventListener('submit', (event) => {
  event.preventDefault();
  const data = Object.fromEntries(new FormData(setup));
  const seed = data.seed.trim();
  delete data.seed;
  if (seed && !/^-?\d+$/.test(seed)) {
    setError('Enter an integer seed.');
    return;
  }
  data.mapSize = Number(data.mapSize);
  const body = JSON.stringify(data);
  selected = null;
  zoom = 1;
  transact(() => request('/reset', seed ? body.slice(0, -1) + ',"seed":' + BigInt(seed).toString() + '}' : body));
});

board.addEventListener('keydown', (event) => {
  const index = Number(event.target.dataset.index);
  if (!Number.isInteger(index) || !match) return;
  const offsets = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -match.size, ArrowDown: match.size };
  const offset = offsets[event.key];
  if (offset == null) return;
  const target = board.querySelector('[data-index="' + (index + offset) + '"]');
  if (target) {
    event.preventDefault();
    event.target.tabIndex = -1;
    target.tabIndex = 0;
    target.focus();
  }
});

byId('zoom-out').onclick = () => { zoom = Math.max(.8, zoom / 1.25); fitBoard(); };
byId('zoom-in').onclick = () => { zoom = Math.min(3, zoom * 1.25); fitBoard(); };
byId('zoom-fit').onclick = () => { zoom = 1; fitBoard(); };
new ResizeObserver(fitBoard).observe(viewport);
byId('retry-bot').onclick = () => transact(() => request('/research/current'));
byId('open').onclick = () => byId('file').click();
byId('file').onchange = async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  const contents = await file.text();
  try {
    const path = JSON.parse(contents).schemaVersion != null ? '/research/replay' : '/research/load';
    selected = null;
    await transact(() => request(path, contents));
  } catch (error) {
    setError(error.message);
  }
};
byId('save').onclick = async () => {
  try {
    const response = await fetch('/research/save');
    if (!response.ok) throw new Error(await response.text());
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = 'polyfish-' + match.seed + '.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    setError(error.message);
  }
};
byId('previous').onclick = () => transact(() => request('/research/replay/seek', { position: match.replay.position - 1 }), false);
byId('next').onclick = () => transact(() => request('/research/replay/seek', { position: match.replay.position + 1 }), false);

try {
  const state = await request('/research/current');
  learnedAvailable = state.learnedAvailable;
  updateLearned();
  if (state.match) await transact(() => Promise.resolve(state));
} catch (error) {
  setError(error.message);
}
