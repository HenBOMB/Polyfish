use polyfish::game::Game;

use polyfish::types::{TerrainType, UnitType};

#[test]
fn test_load_raw_json_parity() {
    let json = r#"{
        "settings": {
            "mode": 1,
            "size": 11,
            "turn": 5,
            "currentPlayerTurnId": 1
        },
        "map": {
            "size": 11,
            "tiles": [
                {
                    "x": 0,
                    "y": 0,
                    "idx": 0,
                    "type": 3,
                    "owner": 1,
                    "hasRoad": true,
                    "unit": {
                        "owner": 1,
                        "type": 3,
                        "hp": 10,
                        "promoted": 1,
                        "xp": 0,
                        "x": 0,
                        "y": 0,
                        "idx": 0
                    }
                },
                {
                    "x": 1,
                    "y": 0,
                    "idx": 1,
                    "type": 4,
                    "owner": 0,
                    "hasRoad": false
                }
            ]
        },
        "tribes": {
            "1": {
                "id": 1,
                "tribe": 7,
                "stars": 10,
                "tech": [1, 2, 4]
            }
        }
    }"#;

    let game = Game::from_json(json).expect("Failed to load JSON");
    let state = &game.state;

    // 1. Verify structural grouping (map wrapper)
    assert_eq!(state.settings.size, 11);
    // This fixture only spells out 2 of the 11x11=121 tiles; `post_load`
    // backfills the rest to default so the dense-tiles invariant holds for
    // every loaded state.
    assert_eq!(state.tiles.len(), 121);

    // 2. Verify Tile mapping (type, hasRoad)
    let tile0 = state.tiles.get(&0).expect("Tile 0 not found");
    assert_eq!(tile0.terrain_type, TerrainType::Field);
    assert!(tile0.has_road);

    // 3. Verify Unit mapping and scaling (hp -> health, promoted -> veteran, xp -> kills)
    let tribe1 = state.tribes.get(&1).expect("Tribe 1 not found");
    assert_eq!(tribe1.units.len(), 1);
    let unit = &tribe1.units[0];
    assert_eq!(unit.unit_type, UnitType::Rider); // Rider is ID 3
    assert_eq!(unit.health, 10.0); // 10 (raw from JSON)
    assert!(unit.veteran);

    // 4. Verify tile owner correctly set from unit
    assert_eq!(tile0._unit_owner_id, Some(1));
}

/// `TileMap`'s custom Serialize/Deserialize must round-trip a real on-disk
/// state byte-for-byte in tile content (EXP_ELO_130 Fix 3: IndexMap -> dense
/// array). Regression guard for the wire format, not just the in-memory type.
#[test]
fn tile_map_round_trips_a_real_saved_state() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("saved_state.json");
    if !path.exists() {
        eprintln!("skipping: {} not present in this checkout", path.display());
        return;
    }
    let game = Game::from_file(&path).expect("Failed to load saved_state.json");
    let json = game.to_json().expect("Should serialize Game");
    let reparsed: polyfish::states::GameState =
        serde_json::from_str(&json).expect("Should deserialize the re-serialized state");

    assert_eq!(reparsed.tiles.len(), game.state.tiles.len());
    for idx in game.state.tiles.keys() {
        let original = game.state.tiles.get(&idx).unwrap();
        let round_tripped = reparsed.tiles.get(&idx).unwrap();
        assert_eq!(original.coords, round_tripped.coords, "tile {idx} coords mismatch");
        assert_eq!(
            original.terrain_type, round_tripped.terrain_type,
            "tile {idx} terrain mismatch"
        );
        assert_eq!(original.owner, round_tripped.owner, "tile {idx} owner mismatch");
    }
}
