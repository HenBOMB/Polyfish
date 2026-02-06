use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::*;

fn main() {
    let settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands, // Pangea might not exist or match Drylands
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed: 6,
        ..Default::default()
    };
    let state = generate(settings);
    let mut capitals = Vec::new();
    for tribe in state.tribes.values() {
        for city in &tribe.cities {
            let (x, y) = (city.tile_index % 11, city.tile_index / 11);
            capitals.push((x, y, city.tile_index));
        }
    }
    println!("Capitals: {:?}", capitals);
    if capitals.len() == 2 {
        let d = (capitals[0].0 - capitals[1].0)
            .abs()
            .max((capitals[0].1 - capitals[1].1).abs());
        println!("Distance: {}", d);
    }

    // List all villages
    for (idx, struct_opt) in &state.structures {
        if let Some(s) = struct_opt {
            if s.structure_type == StructureType::Village {
                println!("Village at ({}, {}) index {}", idx % 11, idx / 11, idx);
            }
        }
    }
}
