use polyfish::game::Game;
use polyfish::states::{GameState, TileState, TribeState};
use polyfish::types::{TerrainType, TribeType};

#[test]
fn test_fow_preservation_in_clone() {
    let mut state = GameState::default();
    state.settings._fow = true; // Enable fog of war
    
    // Player 1
    let mut tribe1 = TribeState::default();
    tribe1.id = 1;
    tribe1.tribe_type = TribeType::Imperius;
    state.tribes.insert(1, tribe1);
    
    // Player 2
    let mut tribe2 = TribeState::default();
    tribe2.id = 2;
    tribe2.tribe_type = TribeType::XinXi;
    state.tribes.insert(2, tribe2);

    // Tile 0: Owned by Player 2, Player 1 has NOT explored it.
    let mut tile0 = TileState::default();
    tile0.terrain_type = TerrainType::Mountain;
    tile0.owner = 2;
    // Explored by Player 2 only
    tile0.explorers.insert(2);
    state.tiles.insert(0, tile0);
    
    // Tile 1: Owned by Player 1, Player 1 HAS explored it.
    let mut tile1 = TileState::default();
    tile1.terrain_type = TerrainType::Field;
    tile1.owner = 1;
    tile1.explorers.insert(1);
    state.tiles.insert(1, tile1);

    let game = Game { state, ..Default::default() };
    
    // Using regular clone() - player 1 can still see player 2's ownership on tile 0
    let cloned_game = game.clone();
    assert_eq!(cloned_game.state.tiles.get(&0).unwrap().owner, 2, "Regular clone should not obscure tile ownership");
    
    // Using clone_for_mcts(1) - player 1 should NOT see player 2's ownership on tile 0
    let fow_game = game.clone_for_mcts(1);
    assert_eq!(fow_game.state.tiles.get(&0).unwrap().owner, 0, "clone_for_mcts should obscure tile ownership for player 1");
    
    // But player 1 CAN see their own tile
    assert_eq!(fow_game.state.tiles.get(&1).unwrap().owner, 1, "clone_for_mcts should retain visible tiles");
}
