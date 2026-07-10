//! Measures capital -> nearest-village distance on training-identical maps
//! (Tiny Drylands, 2 tribes from the training pool), plus Lakes for contrast
//! (Lakes gets the suburb guarantee in mapgen; Drylands does not).
use polyfish::functions::get_chebyshev_distance;
use polyfish::game::Game;
use polyfish::mapgen::{self, MapGenSettings};
use polyfish::types::{MapSize, MapType, StructureType, TribeType};

fn run(map_type: MapType, n: i64, pool: &[TribeType]) {
    let mut hist = [0usize; 12];
    let mut within3 = [0usize; 4]; // 0, 1, 2, 3+ villages at chebyshev <= 3
    let mut caps = 0usize;
    let mut sum_nearest = 0i64;
    let mut total_villages = 0usize;

    for seed in 0..n {
        let t1 = pool[(seed % 5) as usize];
        let mut t2 = pool[((seed / 5) % 5) as usize];
        if t2 == t1 {
            t2 = pool[((seed / 5 + 1) % 5) as usize];
        }
        let mut game = Game::new();
        game.state = mapgen::generate(MapGenSettings {
            size: MapSize::Tiny,
            map_type,
            tribes: vec![t1, t2],
            seed,
            ..Default::default()
        });
        game.post_load();

        // Neutral villages only: capitals keep a Village structure under the
        // city, so exclude any tile owned by a player.
        let villages: Vec<i32> = game
            .state
            .structures
            .iter()
            .filter_map(|(idx, s)| {
                s.as_ref()
                    .filter(|s| s.structure_type == StructureType::Village)
                    .filter(|_| {
                        game.state
                            .tiles
                            .get(idx)
                            .map(|t| t.owner == 0)
                            .unwrap_or(false)
                    })
                    .map(|_| *idx)
            })
            .collect();
        total_villages += villages.len();

        for tribe in game.state.tribes.values() {
            let Some(city) = tribe.cities.first() else {
                continue;
            };
            let nearest = villages
                .iter()
                .map(|&v| get_chebyshev_distance(city.idx, v, 11))
                .min()
                .unwrap_or(99);
            let close = villages
                .iter()
                .filter(|&&v| get_chebyshev_distance(city.idx, v, 11) <= 3)
                .count();
            caps += 1;
            sum_nearest += nearest as i64;
            hist[(nearest as usize).min(11)] += 1;
            within3[close.min(3)] += 1;
        }
    }

    println!("\n=== {map_type:?} (n={n} maps, {caps} capitals) ===");
    println!(
        "villages/map avg: {:.1}",
        total_villages as f64 / n as f64
    );
    println!("nearest-village distance from capital:");
    for (d, &c) in hist.iter().enumerate() {
        if c > 0 {
            println!("  dist {d:2}: {:5.1}%  ({c})", 100.0 * c as f64 / caps as f64);
        }
    }
    println!(
        "  nearest <=3 (turn-4 capture possible): {:.1}%",
        100.0 * hist[..4].iter().sum::<usize>() as f64 / caps as f64
    );
    println!(
        "  nearest <=4 (turn-5 capture possible): {:.1}%",
        100.0 * hist[..5].iter().sum::<usize>() as f64 / caps as f64
    );
    println!("  mean nearest: {:.2}", sum_nearest as f64 / caps as f64);
    println!(
        "villages within chebyshev 3 of capital: 0: {:.1}%  1: {:.1}%  2: {:.1}%  3+: {:.1}%",
        100.0 * within3[0] as f64 / caps as f64,
        100.0 * within3[1] as f64 / caps as f64,
        100.0 * within3[2] as f64 / caps as f64,
        100.0 * within3[3] as f64 / caps as f64
    );
}

fn main() {
    let n: i64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let pool = [
        TribeType::Imperius,
        TribeType::Bardur,
        TribeType::Oumaji,
        TribeType::Kickoo,
        TribeType::XinXi,
    ];
    run(MapType::Drylands, n, &pool);
    run(MapType::Lakes, n, &pool);
}
