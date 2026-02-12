#!/usr/bin/env python3
"""
Infer moves from scraper training data CSVs.

Reads the CSV files produced by steam_replays.py (base state + delta rows),
reconstructs full game states, and classifies each delta as a game move.

Usage:
    python3 infer_moves.py <csv_path> [--output <json_path>]
    python3 infer_moves.py src/scraper/data/training-data/-1124442001.csv
"""

import argparse
import json
import os
import re
import sys
import copy
from typing import Any, Optional

# ============================================================================
# Delta Application (inverse of compute_delta in steam_replays.py)
# ============================================================================

def apply_delta(state: dict, delta: dict) -> dict:
    """
    Apply a delta dict to a full state dict, producing a new state.
    Delta keys are dot-separated paths like 'tribes.2.units[0].coords[0]'
    Values are the new values (None = delete).
    """
    state = copy.deepcopy(state)
    for path, value in delta.items():
        _set_nested(state, path, value)
    return state


def _parse_path(path: str) -> list:
    """Parse a delta path like 'tribes.2.units[0].coords[1]' into segments."""
    segments = []
    for part in path.replace('[', '.[').split('.'):
        if part.startswith('[') and part.endswith(']'):
            idx = part[1:-1]
            segments.append(int(idx))
        else:
            segments.append(part)
    return segments


def _set_nested(obj: Any, path: str, value: Any):
    """Set a value at a nested path in a dict/list structure."""
    # Handle special '.len' suffix (list truncation)
    if path.endswith('.len'):
        container_path = path[:-4]
        segments = _parse_path(container_path)
        container = _navigate(obj, segments)
        if isinstance(container, list) and isinstance(value, int):
            del container[value:]
        return

    segments = _parse_path(path)
    parent = _navigate(obj, segments[:-1])
    key = segments[-1]

    if parent is None:
        return

    if isinstance(parent, dict):
        if value is None:
            parent.pop(str(key), None)
            parent.pop(key, None)
        else:
            parent[str(key) if not isinstance(key, int) else key] = value
    elif isinstance(parent, list):
        if isinstance(key, int):
            # Extend list if needed
            while len(parent) <= key:
                parent.append(None)
            parent[key] = value


def _navigate(obj: Any, segments: list) -> Any:
    """Navigate to a nested location given path segments."""
    current = obj
    for seg in segments:
        if current is None:
            return None
        if isinstance(seg, int):
            if isinstance(current, list) and seg < len(current):
                current = current[seg]
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(str(seg), current.get(seg))
        else:
            return None
    return current


# ============================================================================
# Move Classification
# ============================================================================

class InferredMove:
    def __init__(self, move_type: str, player_id: Optional[int] = None,
                 source_tile: Optional[int] = None, target_tile: Optional[int] = None,
                 option: Optional[str] = None, confidence: float = 0.5,
                 details: Optional[dict] = None):
        self.move_type = move_type
        self.player_id = player_id
        self.source_tile = source_tile
        self.target_tile = target_tile
        self.option = option  # tech type, structure type, unit type, etc.
        self.confidence = confidence
        self.details = details or {}

    def to_dict(self) -> dict:
        d = {
            "move_type": self.move_type,
            "player_id": self.player_id,
            "confidence": round(self.confidence, 2),
        }
        if self.source_tile is not None:
            d["source_tile"] = self.source_tile
        if self.target_tile is not None:
            d["target_tile"] = self.target_tile
        if self.option is not None:
            d["option"] = self.option
        if self.details:
            d["details"] = self.details
        return d


def classify_delta(delta: dict, prev_state: dict, map_size: int) -> list[InferredMove]:
    """
    Classify a delta into one or more inferred moves.
    Returns a list because a single delta could contain multiple 
    collapsed moves (e.g., step + dash attack).
    """
    keys = sorted(delta.keys())
    moves = []

    # Extract per-tribe changes
    tribe_changes = {}  # tribe_id -> {category -> [keys]}
    tile_changes = []
    struct_changes = []
    resource_changes = []

    for k in keys:
        m = re.match(r'^tribes\.(\d+)\.(.+)', k)
        if m:
            tid = int(m.group(1))
            rest = m.group(2)
            if tid not in tribe_changes:
                tribe_changes[tid] = {'units': [], 'cities': [], 'tech': [], 'other': [], 'relations': []}
            if rest.startswith('units'):
                tribe_changes[tid]['units'].append((rest, delta[k]))
            elif rest.startswith('cities'):
                tribe_changes[tid]['cities'].append((rest, delta[k]))
            elif rest.startswith('tech_vanilla'):
                tribe_changes[tid]['tech'].append((rest, delta[k]))
            elif rest.startswith('relations'):
                tribe_changes[tid]['relations'].append((rest, delta[k]))
            else:
                tribe_changes[tid]['other'].append((rest, delta[k]))
        elif k.startswith('tiles.'):
            tile_changes.append((k, delta[k]))
        elif k.startswith('structures.'):
            struct_changes.append((k, delta[k]))
        elif k.startswith('resources.'):
            resource_changes.append((k, delta[k]))

    # --- Check each move type ---

    # 1. END TURN: prevCoords reset + moved/attacked -> False, no coord changes
    for tid, changes in tribe_changes.items():
        unit_keys = [r for r, v in changes['units']]
        has_prev = any('prevCoords' in k for k in unit_keys)
        has_reset = any(('moved' in k or 'attacked' in k) for k in unit_keys)
        has_coord_change = any('coords[' in k and 'prev' not in k for k in unit_keys)

        # Classic end turn: prevCoords + status reset
        if has_prev and has_reset and not has_coord_change:
            moves.append(InferredMove(
                move_type="EndTurn",
                player_id=tid,
                confidence=0.85,
                details={"reason": "prevCoords reset + status clear"}
            ))
        # Variant: only prevCoords updates (no moved/attacked fields present)
        elif has_prev and not has_reset and not has_coord_change:
            all_prevcoords = all('prevCoords' in k for k in unit_keys)
            if all_prevcoords:
                moves.append(InferredMove(
                    move_type="EndTurn",
                    player_id=tid,
                    confidence=0.75,
                    details={"reason": "prevCoords-only reset"}
                ))
        # Variant: attacked/moved reset with production/score but no coords/prevCoords
        elif has_reset and not has_prev and not has_coord_change:
            moves.append(InferredMove(
                move_type="EndTurn",
                player_id=tid,
                confidence=0.70,
                details={"reason": "status reset without prevCoords"}
            ))

    # 2. RESEARCH: tech_vanilla entry added
    for tid, changes in tribe_changes.items():
        for rest, val in changes['tech']:
            if isinstance(val, dict) and val.get('discovered'):
                tech_type = val.get('type')
                moves.append(InferredMove(
                    move_type="Research",
                    player_id=tid,
                    option=f"tech_{tech_type}",
                    confidence=0.95,
                    details={"tech_type_id": tech_type}
                ))

    # 3. STEP: unit coords change + moved=True
    for tid, changes in tribe_changes.items():
        # Group by unit index
        unit_groups = {}
        for rest, val in changes['units']:
            m = re.match(r'units\[(\d+)\]\.(.+)', rest)
            if m:
                u_idx = int(m.group(1))
                field = m.group(2)
                if u_idx not in unit_groups:
                    unit_groups[u_idx] = {}
                unit_groups[u_idx][field] = val

        for u_idx, fields in unit_groups.items():
            # Step detection: coords changed + moved = True
            has_coords = 'coords[0]' in fields or 'coords[1]' in fields
            moved_true = fields.get('moved') is True
            attacked_true = fields.get('attacked') is True

            if has_coords and moved_true and not attacked_true:
                # Determine source and target tiles
                new_x = fields.get('coords[0]')
                new_y = fields.get('coords[1]')

                # Get old coords from prev_state
                old_unit = _get_unit_from_state(prev_state, tid, u_idx)
                old_x = old_unit.get('coords', [None, None])[0] if old_unit else None
                old_y = old_unit.get('coords', [None, None])[1] if old_unit else None

                if new_x is None and old_unit:
                    new_x = old_unit['coords'][0]
                if new_y is None and old_unit:
                    new_y = old_unit['coords'][1]

                source = _coords_to_idx(old_x, old_y, map_size) if old_x is not None and old_y is not None else None
                target = _coords_to_idx(new_x, new_y, map_size) if new_x is not None and new_y is not None else None

                moves.append(InferredMove(
                    move_type="Step",
                    player_id=tid,
                    source_tile=source,
                    target_tile=target,
                    confidence=0.90,
                    details={"unit_idx": u_idx, "from": [old_x, old_y], "to": [new_x, new_y]}
                ))

            # Attack detection: attacked = True (may come with or without coord change)
            elif attacked_true:
                new_x = fields.get('coords[0]')
                new_y = fields.get('coords[1]')
                old_unit = _get_unit_from_state(prev_state, tid, u_idx)

                if old_unit:
                    cur_x = new_x if new_x is not None else old_unit['coords'][0]
                    cur_y = new_y if new_y is not None else old_unit['coords'][1]
                    source = _coords_to_idx(cur_x, cur_y, map_size)
                else:
                    source = None

                # If coords also changed, this is a step+attack (Dash)
                if has_coords:
                    old_x = old_unit['coords'][0] if old_unit else None
                    old_y = old_unit['coords'][1] if old_unit else None
                    step_source = _coords_to_idx(old_x, old_y, map_size) if old_x is not None else None

                    # Try to find the attack target from tile owner changes or health changes
                    attack_target = _find_attack_target(delta, keys, tid, map_size)

                    moves.append(InferredMove(
                        move_type="Step",
                        player_id=tid,
                        source_tile=step_source,
                        target_tile=source,
                        confidence=0.80,
                        details={"unit_idx": u_idx, "dash_attack": True}
                    ))
                    moves.append(InferredMove(
                        move_type="Attack",
                        player_id=tid,
                        source_tile=source,
                        target_tile=attack_target,
                        confidence=0.75,
                        details={"unit_idx": u_idx}
                    ))
                else:
                    attack_target = _find_attack_target(delta, keys, tid, map_size)
                    moves.append(InferredMove(
                        move_type="Attack",
                        player_id=tid,
                        source_tile=source,
                        target_tile=attack_target,
                        confidence=0.80,
                        details={"unit_idx": u_idx}
                    ))

    # 4. BUILD: new structure appears or structure level changes
    for k, v in struct_changes:
        m = re.match(r'structures\.(\d+)', k)
        if m:
            tile_idx = int(m.group(1))
            # New structure (full dict) vs level change
            if isinstance(v, dict):
                moves.append(InferredMove(
                    move_type="Build",
                    player_id=_guess_player(delta, prev_state),
                    target_tile=tile_idx,
                    option=f"struct_{v.get('type')}",
                    confidence=0.90,
                    details={"structure": v}
                ))
            elif 'level' in k:
                moves.append(InferredMove(
                    move_type="Build",
                    player_id=_guess_player(delta, prev_state),
                    target_tile=tile_idx,
                    option=f"struct_level_{v}",
                    confidence=0.80,
                    details={"level_change": v}
                ))

    # 5. HARVEST: resource disappears (None) without a corresponding build
    harvested_tiles = []
    for k, v in resource_changes:
        m = re.match(r'resources\.(\d+)', k)
        if m and v is None:
            tile_idx = int(m.group(1))
            # Check if same delta also has a build on the same tile
            is_build_side_effect = any(
                sk.startswith(f'structures.{tile_idx}') for sk, _ in struct_changes
            )
            if not is_build_side_effect:
                harvested_tiles.append(tile_idx)
                moves.append(InferredMove(
                    move_type="Harvest",
                    player_id=_guess_player(delta, prev_state),
                    target_tile=tile_idx,
                    confidence=0.90,
                    details={"resource_removed": tile_idx}
                ))

    # 6. CAPTURE: tile owner changes + city changes
    owner_changes = []
    for k, v in tile_changes:
        m = re.match(r'tiles\.(\d+)\.owner', k)
        if m and v is not None:
            tile_idx = int(m.group(1))
            new_owner = v
            owner_changes.append((tile_idx, new_owner))

    # If we have city changes AND owner changes, it looks like a capture
    for tid, changes in tribe_changes.items():
        if changes['cities'] and owner_changes:
            # City-related owner changes
            city_tiles = set()
            for rest, val in changes['cities']:
                m = re.match(r'cities\[(\d+)\]\.(.+)', rest)
                if m:
                    city_idx = int(m.group(1))
                    field = m.group(2)
                    if field == 'tileIndex':
                        city_tiles.add(val)

            for tile_idx, new_owner in owner_changes:
                if new_owner == tid:
                    # Check if this is a village capture (rulingCityCoords change)
                    has_ruling = any(k == f'tiles.{tile_idx}.rulingCityCoords' for k, _ in tile_changes)
                    if has_ruling:
                        # This is territory expansion from a capture, not the capture itself
                        continue

            # If cities list gained new entries, it's a capture
            for rest, val in changes['cities']:
                if re.match(r'cities\[(\d+)\]$', rest) and isinstance(val, dict):
                    moves.append(InferredMove(
                        move_type="Capture",
                        player_id=tid,
                        target_tile=val.get('tileIndex'),
                        confidence=0.85,
                        details={"city": val.get('name')}
                    ))

    # 7. REWARD: city rewards change
    for tid, changes in tribe_changes.items():
        for rest, val in changes['cities']:
            if 'rewards' in rest and val is not None:
                m = re.match(r'cities\[(\d+)\]\.rewards\[(\d+)\]', rest)
                if m:
                    moves.append(InferredMove(
                        move_type="Reward",
                        player_id=tid,
                        option=f"reward_{val}",
                        confidence=0.90,
                        details={"reward_type_id": val}
                    ))

    # 8. SUMMON: new unit appears (units array grows)
    for tid, changes in tribe_changes.items():
        for rest, val in changes['units']:
            if re.match(r'units\[\d+\]$', rest) and isinstance(val, dict):
                moves.append(InferredMove(
                    move_type="Summon",
                    player_id=tid,
                    target_tile=_coords_to_idx(val.get('coords', [None])[0],
                                                val.get('coords', [None, None])[1],
                                                map_size) if val.get('coords') else None,
                    option=f"unit_{val.get('type')}",
                    confidence=0.85,
                    details={"unit_type_id": val.get('type')}
                ))

    # 8b. UNIT KILLED: units.len decreases (unit array truncated)
    for tid, changes in tribe_changes.items():
        for rest, val in changes['units']:
            if rest == 'units.len' and isinstance(val, int):
                moves.append(InferredMove(
                    move_type="UnitKilled",
                    player_id=tid,
                    confidence=0.80,
                    details={"new_unit_count": val, "reason": "units array truncated"}
                ))

    # 8c. GAME END: resignedTurn or killedTurn changes
    for tid, changes in tribe_changes.items():
        for rest, val in changes['other']:
            if rest == 'resignedTurn' and isinstance(val, int) and val > 0:
                moves.append(InferredMove(
                    move_type="GameEnd",
                    player_id=tid,
                    confidence=0.95,
                    details={"reason": "resigned", "turn": val}
                ))
            elif rest == 'killedTurn' and isinstance(val, int) and val > 0:
                moves.append(InferredMove(
                    move_type="GameEnd",
                    player_id=tid,
                    confidence=0.95,
                    details={"reason": "killed", "turn": val}
                ))
            elif rest == 'bot' and val is True:
                # Player disconnected / became bot
                pass

    # 9. RECOVER / DAMAGE: health changes without attack context
    for tid, changes in tribe_changes.items():
        for rest, val in changes['units']:
            if 'health' in rest and isinstance(val, (int, float)):
                old_unit_idx = re.match(r'units\[(\d+)\]', rest)
                if old_unit_idx:
                    u_idx = int(old_unit_idx.group(1))
                    old_unit = _get_unit_from_state(prev_state, tid, u_idx)
                    old_hp = old_unit.get('health', 0) if old_unit else 0
                    unit_coords = old_unit.get('coords', [None, None]) if old_unit else [None, None]
                    tile = _coords_to_idx(unit_coords[0], unit_coords[1], map_size)

                    if val > old_hp:
                        # Health increased → Recover
                        if not any(m.move_type in ('Attack', 'Step') for m in moves):
                            moves.append(InferredMove(
                                move_type="Recover",
                                player_id=tid,
                                source_tile=tile,
                                confidence=0.70,
                                details={"unit_idx": u_idx, "old_hp": old_hp, "new_hp": val}
                            ))
                    elif val < old_hp:
                        # Health decreased → took damage (attack result on THIS unit)
                        # Only add if not already covered by an Attack move
                        already_covered = any(
                            m.move_type == 'Attack' and m.player_id != tid
                            for m in moves
                        )
                        if not already_covered:
                            moves.append(InferredMove(
                                move_type="Damage",
                                player_id=tid,
                                source_tile=tile,
                                confidence=0.65,
                                details={"unit_idx": u_idx, "old_hp": old_hp, "new_hp": val,
                                         "reason": "health decreased (attacked by opponent)"}
                            ))

    # --- Noise filtering ---
    # If we have NO moves but only explorer/score changes, mark as noise
    if not moves:
        only_noise = all(
            ('explorers' in k or '_unitOwnerID' in k or 'score' in k or 'stars' in k
             or 'production' in k or 'population' in k or 'progress' in k)
            for k in keys
        )
        if only_noise:
            moves.append(InferredMove(
                move_type="Noise",
                confidence=0.50,
                details={"reason": "Only explorer/score/production updates", "keys": keys[:5]}
            ))
        else:
            # Last resort: check for coord changes without moved flag (unit list reshuffled after kill)
            has_any_coords = any('coords[' in k and 'prev' not in k for k in keys)
            has_any_prevcoords = any('prevCoords' in k for k in keys)
            if has_any_coords and has_any_prevcoords:
                moves.append(InferredMove(
                    move_type="UnitListReshuffle",
                    confidence=0.60,
                    details={"reason": "coords+prevCoords shift (unit killed, list reindexed)", "keys": keys[:10]}
                ))
            else:
                moves.append(InferredMove(
                    move_type="Unknown",
                    confidence=0.30,
                    details={"keys": keys[:10]}
                ))

    return moves


def _get_unit_from_state(state: dict, tribe_id: int, unit_idx: int) -> Optional[dict]:
    """Get a unit dict from the full state."""
    tribes = state.get('tribes', {})
    tribe = tribes.get(str(tribe_id))
    if not tribe:
        return None
    units = tribe.get('units', [])
    if unit_idx < len(units):
        return units[unit_idx]
    return None


def _coords_to_idx(x, y, map_size: int) -> Optional[int]:
    """Convert (x,y) coords to tile index."""
    if x is None or y is None:
        return None
    return int(y) * map_size + int(x)


def _idx_to_coords(idx: int, map_size: int) -> tuple[int, int]:
    """Convert tile index to (x, y) coords."""
    return idx % map_size, idx // map_size


def _find_attack_target(delta: dict, keys: list, attacker_tribe_id: int, map_size: int) -> Optional[int]:
    """Try to find the attack target from delta signals."""
    # Look for enemy health changes
    for k in keys:
        m = re.match(r'tribes\.(\d+)\.units\[\d+\]\.health', k)
        if m:
            tid = int(m.group(1))
            if tid != attacker_tribe_id:
                # Find coords of this enemy unit
                u_match = re.match(r'tribes\.(\d+)\.units\[(\d+)\]', k)
                if u_match:
                    # We'd need the prev_state to find coords, but we don't have it here
                    pass

    # Look for _unitOwnerID changes (unit killed)
    for k in keys:
        m = re.match(r'tiles\.(\d+)\._unitOwnerID', k)
        if m:
            tile_idx = int(m.group(1))
            val = delta[k]
            if val is None:
                # A unit was removed from this tile — likely killed
                return tile_idx
    return None


def _guess_player(delta: dict, prev_state: dict) -> Optional[int]:
    """Guess which player performed this action from delta changes."""
    for k in delta:
        m = re.match(r'tribes\.(\d+)\.', k)
        if m:
            tid = int(m.group(1))
            # Prefer tribe with stars decrease (they spent something)
            if 'stars' in k:
                return tid
    # Fallback
    for k in delta:
        m = re.match(r'tribes\.(\d+)\.', k)
        if m:
            return int(m.group(1))
    return None


# ============================================================================
# CSV Processing
# ============================================================================

def read_training_csv(path: str) -> tuple[dict, list[tuple[int, str, dict]]]:
    """
    Read a training CSV, returning (base_state, [(outcome, type, data), ...]).
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    rows = []
    base_state = None

    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',', 2)
        if len(parts) < 3:
            continue
        outcome = int(parts[0])
        dtype = parts[1]
        data = json.loads(parts[2])
        rows.append((outcome, dtype, data))

        if dtype == 'base' and base_state is None:
            base_state = data

    return base_state, rows


def process_game(csv_path: str) -> list[dict]:
    """
    Process a single game CSV and return a list of inferred move dicts.
    """
    base_state, rows = read_training_csv(csv_path)
    if not base_state:
        print(f"ERROR: No base state found in {csv_path}", file=sys.stderr)
        return []

    # Detect map size from tile count
    tiles = base_state.get('tiles', {})
    tile_count = len(tiles)
    map_size = int(tile_count ** 0.5)
    if map_size * map_size != tile_count:
        # Try from first tile coords
        max_coord = 0
        for tid, t in tiles.items():
            coords = t.get('coords', [0, 0])
            max_coord = max(max_coord, max(coords[0], coords[1]))
        map_size = max_coord + 1

    print(f"Map size: {map_size}x{map_size} ({tile_count} tiles)")
    print(f"Total rows: {len(rows)} ({sum(1 for _, t, _ in rows if t == 'base')} base, "
          f"{sum(1 for _, t, _ in rows if t == 'delta')} delta)")

    # Reconstruct states and classify
    current_state = copy.deepcopy(base_state)
    results = []
    move_idx = 0

    for i, (outcome, dtype, data) in enumerate(rows):
        if dtype == 'base':
            current_state = copy.deepcopy(data)
            continue

        # Apply delta to reconstruct next state
        prev_state = current_state
        next_state = apply_delta(current_state, data)

        # Classify the delta
        inferred = classify_delta(data, prev_state, map_size)

        for m in inferred:
            result = m.to_dict()
            result["row_index"] = i + 1  # 1-indexed (header = 0)
            result["outcome"] = outcome
            result["move_index"] = move_idx
            result["delta_key_count"] = len(data)
            results.append(result)
            move_idx += 1

        current_state = next_state

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Infer moves from scraper training data CSVs")
    parser.add_argument('csv_path', help='Path to training CSV file')
    parser.add_argument('-o', '--output', help='Output JSON path (default: <csv_basename>_moves.json)')
    parser.add_argument('--summary', action='store_true', help='Print move type summary')
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    results = process_game(args.csv_path)

    # Output
    output_path = args.output
    if not output_path:
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        output_dir = os.path.dirname(args.csv_path) or '.'
        output_path = os.path.join(output_dir, f"{base}_moves.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} inferred moves to {output_path}")

    # Summary
    if args.summary or True:  # Always show summary for now
        from collections import Counter
        type_counts = Counter(r['move_type'] for r in results)
        print("\n--- Move Type Summary ---")
        for mt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            avg_conf = sum(r['confidence'] for r in results if r['move_type'] == mt) / count
            print(f"  {mt:12s}: {count:3d} (avg conf: {avg_conf:.2f})")

        # Print low-confidence moves
        low_conf = [r for r in results if r['confidence'] < 0.5]
        if low_conf:
            print(f"\n--- {len(low_conf)} Low Confidence Moves ---")
            for r in low_conf[:10]:
                print(f"  Row {r['row_index']:3d}: {r['move_type']:12s} "
                      f"(conf={r['confidence']:.2f}) {r.get('details', {})}")


if __name__ == '__main__':
    main()
