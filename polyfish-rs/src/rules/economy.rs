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
