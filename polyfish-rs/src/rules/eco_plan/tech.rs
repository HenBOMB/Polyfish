//! Tech-chain and cost accounting for the economy planner: which techs
//! a lane needs, what they bill against what's owned, and the structure
//! each hub eats.

use super::*;
use crate::settings::technology::{get_tech_cost, get_technology_setting};
use crate::types::StructureType;
use std::collections::HashSet;

/// Techs a lane needs, in dependency order. Returned regardless of endowment;
/// the caller prices only the ones not already owned.
pub fn lane_chain(lane: Lane, convert: bool) -> Vec<TechnologyType> {
    // Derived from the structures the lane actually places, so adding a lane
    // cannot forget to price its techs. Everything else a build-out touches is
    // billed per-buy; only the terrain conversions are extra.
    let mut v = structure_techs(lane_partner_type(lane));
    v.extend(structure_techs(lane_hub(lane).0));
    if convert {
        match lane {
            // GrowForest: Field -> Forest. Its prerequisite is pulled in by
            // `tech_bill`, so only the ability tech is named.
            Lane::Forest => v.push(TechnologyType::Spiritualism),
            // BurnForest rides along with Construction, already in the chain.
            Lane::Farm | Lane::Mine => {}
        }
    }
    v
}

/// Market needs Trade, which is three techs deep off nothing the economy lanes
/// buy. Charged once empire-wide, and only when a Market is actually placed.
pub fn market_chain() -> [TechnologyType; 3] {
    [
        TechnologyType::Riding,
        TechnologyType::Roads,
        TechnologyType::Trade,
    ]
}

pub fn tech_bill(chain: &[TechnologyType], owned: &HashSet<TechnologyType>, cities: i32) -> i32 {
    tech_bill_itemised(chain, owned, cities).0
}

/// The bill, plus the techs actually charged for — the build card needs to name
/// them, and deriving the list separately would let it drift from the price.
pub fn tech_bill_itemised(
    chain: &[TechnologyType],
    owned: &HashSet<TechnologyType>,
    cities: i32,
) -> (i32, Vec<TechnologyType>) {
    let mut bought = Vec::new();
    let mut bill = 0;
    let mut have = owned.clone();
    for t in chain {
        if have.contains(t) {
            continue;
        }
        // Pull in any unowned prerequisite too.
        let mut stack = vec![*t];
        while let Some(cur) = stack.pop() {
            if have.contains(&cur) {
                continue;
            }
            let s = get_technology_setting(cur);
            if let Some(req) = s.requires {
                if !have.contains(&req) {
                    stack.push(cur);
                    stack.push(req);
                    continue;
                }
            }
            // Price each tech at ITS OWN tier. Billing prerequisites at the
            // target's tier over-charged every deep chain.
            bill += get_tech_cost(
                cities,
                crate::settings::technology::tech_tier(cur),
                false,
            );
            bought.push(cur);
            have.insert(cur);
        }
    }
    (bill, bought)
}

/// The tech a structure needs, read off the tech table via the engine's own
/// reverse lookup so this cannot drift from it.
pub fn structure_techs(s: StructureType) -> Vec<TechnologyType> {
    crate::settings::technology::get_tech_unlocking_structure(s)
        .into_iter()
        .collect()
}

/// The partner structure a lane's hub eats, and the hub itself.
pub fn lane_hub(lane: Lane) -> (StructureType, &'static str) {
    match lane {
        Lane::Forest => (StructureType::Sawmill, "LumberHut"),
        Lane::Farm => (StructureType::Windmill, "Farm"),
        Lane::Mine => (StructureType::Forge, "Mine"),
    }
}

pub fn lane_partner_type(lane: Lane) -> StructureType {
    match lane {
        Lane::Forest => StructureType::LumberHut,
        Lane::Farm => StructureType::Farm,
        Lane::Mine => StructureType::Mine,
    }
}
