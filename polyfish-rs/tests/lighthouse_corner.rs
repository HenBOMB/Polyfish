use polyfish::actions::set_visible_tiles;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, StructureType, TribeType};

#[test]
fn test_lighthouse_corner_placement() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // 11x11
    settings.map_type = MapType::Drylands;
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 12345;

    let state = generate(settings);
    let size = state.settings.size;
    let corners = [0, size - 1, size * (size - 1), size * size - 1];

    for &idx in &corners {
        let structure = state
            .structures
            .get(&idx)
            .expect("Should have structure at corner");
        let s = structure.as_ref().expect("Should have structure data");
        assert_eq!(
            s.structure_type,
            StructureType::Lighthouse,
            "Corner at {} should be a lighthouse",
            idx
        );
    }
}

#[test]
fn test_lighthouse_corner_visibility_hidden() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // 11x11
    settings.map_type = MapType::Drylands;
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 6789; // Use specific seed that places capital near edge but not on corner

    let mut state = generate(settings);
    let size = state.settings.size;
    let corners = [0, size - 1, size * (size - 1), size * size - 1];

    // Imperius is Tribe 1
    let _ = set_visible_tiles(&mut state, 1);

    // Find where Tribe 1's city is
    let city_idx = state.tribes[&1].cities[0].tile_index;

    for &idx in &corners {
        if idx == city_idx {
            continue; // We expect to see our own city
        }

        // If the corner is within distance 2 of our city, it would normally be visible (5x5)
        let cx = city_idx % size;
        let cy = city_idx / size;
        let kx = idx % size;
        let ky = idx / size;
        let dist = (cx - kx).abs().max((cy - ky).abs());

        if dist <= 2 {
            assert!(
                !state._visible_tiles.contains_key(&idx),
                "Corner {} at distance {} should be hidden from initial city vision",
                idx,
                dist
            );
        }
    }
}
