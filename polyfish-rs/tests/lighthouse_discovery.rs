use polyfish::actions::discovery::discover_tiles;
use polyfish::functions::get_city_at;
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, StructureType, TribeType};

#[test]
fn test_lighthouse_discovery_rewards_population() {
    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // 11x11
    settings.map_type = MapType::Drylands; // Ensures corners are lighthouses
    settings.tribes = vec![TribeType::Imperius, TribeType::Bardur];
    settings.seed = 12345;

    let mut state = generate(settings);
    let size = state.settings.size;

    // Find a corner lighthouse
    let lh_idx = 0; // Top-left corner
    let structure = state
        .structures
        .get(&lh_idx)
        .expect("Should have structure at corner")
        .as_ref()
        .expect("Should have structure data");
    assert_eq!(structure.structure_type, StructureType::Lighthouse);

    // Get Tribe 1 (Imperius) capital
    let tribe_id = 1;
    let capital_idx = state.tribes[&tribe_id].cities[0].tile_index;
    let initial_pop = state.tribes[&tribe_id].cities[0].population;

    // Ensure lighthouse is not explored by Tribe 1
    let tile = state.tiles.get(&lh_idx).unwrap();
    assert!(!tile.explorers.contains(&tribe_id));

    // Discover the lighthouse
    let _ = discover_tiles(&mut state, tribe_id, None, Some(vec![lh_idx]));

    // Check if explored
    let tile = state.tiles.get(&lh_idx).unwrap();
    assert!(
        tile.explorers.contains(&tribe_id),
        "Lighthouse should be marked explored"
    );

    // Check capital population
    let new_pop = state.tribes[&tribe_id].cities[0].population;
    assert_eq!(
        new_pop,
        initial_pop + 1,
        "Discovering lighthouse should increase capital population by 1"
    );
}
