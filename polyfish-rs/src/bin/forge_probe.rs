//! Temporary probe #2: can our Tiny maps reproduce the user's iPad scenario —
//! Xin-xi capital Forge level 3 with NO border growth (all in the initial 3x3),
//! level 4 after border growth (territory Cheb<=2)? Delete after investigation.

use polyfish::functions::{get_chebyshev_distance, get_square_indices};
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{
    MapSize, MapType, ResourceType, StructureType, TerrainType, TribeType,
};

fn main() {
    let n_maps: u32 = 2000;
    let mut ring1 = [0u32; 9]; // metal count at Cheb<=1 of capital (8 ring tiles)
    let mut lvl3_no_bg = 0u32;
    let mut lvl4_bg = 0u32;
    let mut both = 0u32;
    let mut best_no_bg_sum = 0u64;
    let mut best_bg_sum = 0u64;
    let mut village2 = 0u32; // maps with a non-capital village owning a lvl>=2 forge in its 3x3

    for seed in 0..n_maps {
        let state = generate(MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::XinXi, TribeType::Imperius],
            seed: 950_000 + seed as i64,
            version: 115,
        });
        let size = state.settings.size;
        let is_metal = |idx: &i32| {
            matches!(
                state.resources.get(idx),
                Some(Some(r)) if r.resource_type == ResourceType::Metal
            )
        };
        let cap = state
            .tiles
            .values()
            .find(|t| t.capital_of == 1)
            .map(|t| t.coords.idx)
            .expect("capital");

        let m1 = get_square_indices(cap, 1, size)
            .into_iter()
            .filter(|i| *i != cap && is_metal(i))
            .count();
        ring1[m1.min(8)] += 1;

        // Best forge level with territory radius r: site Field/Forest, not the
        // capital/village tile, within Cheb<=r; mines = metal within Cheb<=r.
        let best_at = |r: i32| -> usize {
            let mut best = 0usize;
            for (idx, tile) in &state.tiles {
                if *idx == cap
                    || get_chebyshev_distance(*idx, cap, size) > r
                    || !matches!(
                        tile.terrain_type,
                        TerrainType::Field | TerrainType::Forest
                    )
                    || matches!(
                        state.structures.get(idx),
                        Some(Some(s)) if s.structure_type == StructureType::Village
                    )
                {
                    continue;
                }
                let lvl = get_square_indices(*idx, 1, size)
                    .into_iter()
                    .filter(|n| {
                        n != idx
                            && is_metal(n)
                            && get_chebyshev_distance(*n, cap, size) <= r
                    })
                    .count();
                best = best.max(lvl);
            }
            best
        };
        let b1 = best_at(1);
        let b2 = best_at(2);
        best_no_bg_sum += b1 as u64;
        best_bg_sum += b2 as u64;
        if b1 >= 3 {
            lvl3_no_bg += 1;
        }
        if b2 >= 4 {
            lvl4_bg += 1;
        }
        if b1 >= 3 && b2 >= 4 {
            both += 1;
        }

        // Second-city check: any non-capital village whose own 3x3 supports lvl>=2
        let mut v2 = false;
        for (vidx, s) in &state.structures {
            if !matches!(s, Some(s) if s.structure_type == StructureType::Village) {
                continue;
            }
            for site in get_square_indices(*vidx, 1, size) {
                if site == *vidx {
                    continue;
                }
                let Some(t) = state.tiles.get(&site) else { continue };
                if !matches!(t.terrain_type, TerrainType::Field | TerrainType::Forest) {
                    continue;
                }
                let lvl = get_square_indices(site, 1, size)
                    .into_iter()
                    .filter(|n| {
                        *n != site
                            && is_metal(n)
                            && get_chebyshev_distance(*n, *vidx, size) <= 1
                    })
                    .count();
                if lvl >= 2 {
                    v2 = true;
                }
            }
        }
        if v2 {
            village2 += 1;
        }
    }

    let pct = |c: u32| 100.0 * c as f64 / n_maps as f64;
    println!("Tiny11 Drylands XinXi+Imperius, n={n_maps}");
    print!("  metal in capital ring (Cheb<=1): ");
    for (k, c) in ring1.iter().enumerate() {
        if *c > 0 {
            print!("{k}:{:.1}% ", pct(*c));
        }
    }
    println!();
    println!(
        "  best forge, no border growth: mean {:.2} | >=3 {:.1}%",
        best_no_bg_sum as f64 / n_maps as f64,
        pct(lvl3_no_bg)
    );
    println!(
        "  best forge, with border growth (Cheb<=2): mean {:.2} | >=4 {:.1}%",
        best_bg_sum as f64 / n_maps as f64,
        pct(lvl4_bg)
    );
    println!(
        "  user scenario (lvl3 no-BG AND lvl4 with BG): {:.1}% | squared (2 games back-to-back): {:.2}%",
        pct(both),
        pct(both) * pct(both) / 100.0
    );
    println!("  some other village supports lvl>=2 in own 3x3: {:.1}%", pct(village2));
}
