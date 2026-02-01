use crate::states::{GameState, PlayerId};
use crate::types::{TerrainType, UnitType};
use candle_core::{Device, Result, Tensor};

pub const MAP_HEIGHT: usize = 30;
pub const MAP_WIDTH: usize = 30;

// Feature Channels
// 0: Terrain: None/Ground
// 1: Terrain: Water/Ocean
// 2: Terrain: Forest
// 3: Terrain: Mountain
// 4: Terrain: Field
// 5: Has Road
// 6: Has Building
// 7-19: Unit Types
// 20: Unit Owner
// 21: Unit HP
// 22: City Owner
// 23: City Level
// 24: Visibility (1.0)
// 25: Turn
// 26: Stars
pub const NUM_CHANNELS: usize = 27;

pub fn state_to_tensor(state: &GameState, perspective: PlayerId) -> Result<Tensor> {
    let mut data = vec![0.0f32; NUM_CHANNELS * MAP_HEIGHT * MAP_WIDTH];
    let map_size = state.settings.size as usize;

    for y in 0..map_size {
        for x in 0..map_size {
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }
            let idx = (y * map_size + x) as i32;

            if let Some(tile) = state.tiles.get(&idx) {
                // Terrain
                let t_idx = match tile.terrain_type {
                    TerrainType::Water | TerrainType::Ocean => 1,
                    TerrainType::Forest => 2,
                    TerrainType::Mountain => 3,
                    TerrainType::Field => 4,
                    TerrainType::Ice => 1, // Treat ice as water-like for now? Or separate?
                    _ => 0,                // None/Ground
                };
                set_feat(&mut data, t_idx, x, y, 1.0);

                if tile.has_road {
                    set_feat(&mut data, 5, x, y, 1.0);
                }

                // Buildings
                if let Some(Some(_structure)) = state.structures.get(&idx) {
                    set_feat(&mut data, 6, x, y, 1.0);
                }
            }
        }
    }

    for (player_id, tribe) in &state.tribes {
        for unit in &tribe.units {
            let x = unit.coords.x as usize;
            let y = unit.coords.y as usize;
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }

            // Mapping common units to 7-19 (12 slots)
            let u_idx = match unit.unit_type {
                UnitType::Warrior => 0,
                UnitType::Rider => 1,
                UnitType::Archer => 2,
                UnitType::Defender => 3,
                UnitType::Swordsman => 4,
                UnitType::Catapult => 5,
                UnitType::Knight => 6,
                UnitType::MindBender => 7,
                UnitType::Giant => 8, // Flattening super units slightly
                UnitType::Crab | UnitType::Gaami | UnitType::Centipede => 8,
                // Naval
                UnitType::Dinghy | UnitType::Raft | UnitType::Scout => 9,
                UnitType::Rammer | UnitType::Bomber => 10,
                UnitType::Juggernaut | UnitType::Pirate => 11,
                _ => 12,
            };
            if u_idx < 13 {
                set_feat(&mut data, 7 + u_idx, x, y, 1.0);
            } else {
                // Overflow bucket
                set_feat(&mut data, 19, x, y, 1.0);
            }

            if *player_id == perspective {
                set_feat(&mut data, 20, x, y, 1.0);
            } else {
                set_feat(&mut data, 20, x, y, -1.0);
            }

            set_feat(&mut data, 21, x, y, unit.health as f32 / 40.0);
        }

        for city in &tribe.cities {
            let idx = city.tile_index;
            let x = (idx % state.settings.size) as usize;
            let y = (idx / state.settings.size) as usize;
            if x >= MAP_WIDTH || y >= MAP_HEIGHT {
                continue;
            }

            if *player_id == perspective {
                set_feat(&mut data, 22, x, y, 1.0);
            } else {
                set_feat(&mut data, 22, x, y, -1.0);
            }
            set_feat(&mut data, 23, x, y, city.level as f32 / 10.0);
        }
    }

    for y in 0..MAP_HEIGHT {
        for x in 0..MAP_WIDTH {
            set_feat(&mut data, 24, x, y, 1.0); // Full vis assumption
            set_feat(&mut data, 25, x, y, state.settings.turn as f32 / 50.0);
            if let Some(tribe) = state.tribes.get(&perspective) {
                set_feat(&mut data, 26, x, y, tribe.stars as f32 / 100.0);
            }
        }
    }

    Tensor::from_vec(data, (1, NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH), &Device::Cpu)
}

fn set_feat(data: &mut Vec<f32>, channel: usize, x: usize, y: usize, val: f32) {
    let idx = channel * (MAP_HEIGHT * MAP_WIDTH) + (y * MAP_WIDTH + x);
    if idx < data.len() {
        data[idx] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Game;

    #[test]
    fn test_tensor_shape() {
        let game = Game::default();
        let tensor = state_to_tensor(&game.state, 1).unwrap();
        let dims = tensor.dims();
        assert_eq!(dims, &[1, NUM_CHANNELS, MAP_HEIGHT, MAP_WIDTH]);
    }
}
