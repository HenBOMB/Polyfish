//! Economy rules — one implementation of adjacency, levels and territory.

use crate::settings::structures::get_structure_setting;
use crate::states::{CityState, GameState, PlayerId};
use crate::types::{ResourceType, StructureType};
use std::collections::HashSet;

/// A Market's own level, and so its income, is capped here.
pub const MARKET_MAX_LEVEL: i32 = 8;

/// Pop needed to take a city from level `from` to level `to`.
/// The engine grants a level when `progress >= level + 1` (`actions/city.rs`).
pub fn pop_to_reach(from: i32, to: i32) -> i32 {
    (from..to).map(|l| l + 1).sum()
}

/// Highest level reachable with `pop`, starting from level 1.
pub fn level_at_pop(pop: i32) -> i32 {
    let mut level = 1;
    while pop_to_reach(1, level + 1) <= pop {
        level += 1;
    }
    level
}

/// Super units available at a level: every reward slot from 4 up offers one.
pub fn super_units_at_level(level: i32) -> i32 {
    (level - 4).max(0)
}

/// Does a structure of this type claim the tile it stands on?
///
/// Road is the one that does not: `create_structure` stores it as a tile flag
/// instead of an occupant, and it may share a tile with a real structure.
pub fn occupies_tile(struct_type: StructureType) -> bool {
    struct_type != StructureType::Road
}

/// The structure that WORKS `resource` rather than crushing it — a Farm keeps
/// its Crop, a Mine its Metal. The pairing lives in `settings/resources.rs`;
/// `StructureSetting.resource_type` is the same fact read the other way, and
/// `tests/rules_ssot.rs` holds the two to each other.
pub fn worker_structure(resource: ResourceType) -> Option<StructureType> {
    crate::settings::resources::get_resource_setting(resource).struct_required
}

/// Does building `struct_type` over `resource` destroy it?
///
/// Build legality deliberately ignores resources (`moves/build.rs` checks
/// terrain and occupancy only), so an *undeveloped* Crop or Fruit field is a
/// legal site for any terrain structure and the build crushes what stands
/// there. A *developed* tile is protected by the occupancy check instead — you
/// cannot build over a Farm, only destroy it first. The exception is the
/// structure that works the resource, which is why it is allowed on that tile
/// at all.
pub fn build_consumes_resource(struct_type: StructureType, resource: ResourceType) -> bool {
    occupies_tile(struct_type) && worker_structure(resource) != Some(struct_type)
}

/// How many friendly partners feed the adjacency structure on `idx`.
///
/// This is the quantity `build_structure` multiplies `reward_pop` by and the
/// quantity a Market reads as a hub's "level". Ownership is **player**-scoped
/// throughout the engine, so a partner across a city border still counts.
///
/// Core form: caller supplies the partner set, so the build path — which already
/// holds the structure's settings — pays nothing extra.
pub fn partner_count_with(
    state: &GameState,
    idx: i32,
    partners: &HashSet<StructureType>,
    owner: PlayerId,
) -> i32 {
    if partners.is_empty() || owner == 0 {
        return 0;
    }
    crate::functions::get_adjacent_indices(state, idx, 1)
        .into_iter()
        .filter(|&adj| {
            state.tiles.get(&adj).is_some_and(|t| t.owner == owner)
                && crate::functions::get_structure_at(state, adj)
                    .is_some_and(|s| partners.contains(&s.structure_type))
        })
        .count() as i32
}

/// How many friendly partners feed a hub of `hub_type` sited on `idx`.
pub fn partner_count(
    state: &GameState,
    idx: i32,
    hub_type: StructureType,
    owner: PlayerId,
) -> i32 {
    partner_count_with(
        state,
        idx,
        &get_structure_setting(hub_type).adjacent_types,
        owner,
    )
}

/// Partners this site could EVER collect — the ceiling, not today's count.
///
/// `partner_count` answers "what feeds this hub now", which is the wrong
/// question at the instant a hub is placed: its partners are bought over the
/// following turns, and the search horizon (~7 plies, less than one game turn)
/// never reaches them. Two tiles that will end up with five partners and one
/// look identical to a realized count. The ceiling makes a site's future
/// legible immediately, and it is a function of state alone, so a potential
/// built on it stays policy-invariant.
///
/// A tile counts when it already holds a partner, or could: the partner's own
/// terrain rule, on an unoccupied tile, and — for partners that WORK a
/// resource (Farm on Crop, Mine on Metal) — only where that resource still
/// stands. Terrain the owner could convert (burning forest to field for a
/// Farm) is deliberately excluded: that is a plan, not a property of the tile.
pub fn partner_ceiling_with(
    state: &GameState,
    idx: i32,
    partners: &HashSet<StructureType>,
    owner: PlayerId,
) -> i32 {
    if partners.is_empty() || owner == 0 {
        return 0;
    }
    crate::functions::get_adjacent_indices(state, idx, 1)
        .into_iter()
        .filter(|&adj| {
            let Some(tile) = state.tiles.get(&adj) else {
                return false;
            };
            if tile.owner != owner {
                return false;
            }
            match crate::functions::get_structure_at(state, adj) {
                // The ceiling includes what is already realized.
                Some(s) => partners.contains(&s.structure_type),
                None => partners.iter().any(|&p| {
                    let ps = get_structure_setting(p);
                    if !ps.terrain_types.contains(&tile.terrain_type) {
                        return false;
                    }
                    match ps.resource_type {
                        // Read the map, not `get_resource_at`: that filters by
                        // the acting player's tech visibility, and a potential
                        // must not depend on whose turn it is.
                        Some(r) => {
                            state.resources.get(&adj).and_then(|x| x.as_ref())
                                .map(|x| x.resource_type) == Some(r)
                        }
                        None => true,
                    }
                }),
            }
        })
        .count() as i32
}

/// Partners a hub of `hub_type` sited on `idx` could ever collect.
pub fn partner_ceiling(
    state: &GameState,
    idx: i32,
    hub_type: StructureType,
    owner: PlayerId,
) -> i32 {
    partner_ceiling_with(
        state,
        idx,
        &get_structure_setting(hub_type).adjacent_types,
        owner,
    )
}

/// Do these two tiles answer to the same city?
///
/// Hubs are `limited_per_city`, so the alternatives to a hub placement are the
/// tiles of the SAME city. A better site under a different city is not an
/// option forgone — that city can still build its own hub there. Comparing
/// across cities makes a correct placement look wrong: it flagged 25 of 34
/// "sub-optimal" hub placements that were nothing of the kind (Aug 2026).
pub fn same_city(state: &GameState, a: i32, b: i32) -> bool {
    let city_of = |t: i32| crate::functions::get_city_owning_tile(state, t).map(|c| c.idx);
    match (city_of(a), city_of(b)) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Tiles this city actually rules.
///
/// `_territory` is the radius-2 square filtered by *player* ownership, so a tile
/// two away from two of your own cities appears in **both** lists — while
/// `ruling_city_coords` says it belongs to exactly one. Consumers that sum over
/// raw `_territory` double-count those tiles; iterate this instead.
pub fn territory_tiles<'a>(
    state: &'a GameState,
    city: &'a CityState,
) -> impl Iterator<Item = i32> + 'a {
    city._territory.iter().copied().filter(move |idx| {
        state
            .tiles
            .get(idx)
            .and_then(|t| t.ruling_city_coords.as_ref())
            .is_none_or(|rc| rc.idx == city.idx)
    })
}
