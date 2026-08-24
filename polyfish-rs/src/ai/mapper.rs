use crate::ai::network::NUM_ACTION_TYPES;
use crate::moves::Move;
use crate::types::*;
use std::collections::HashMap;
use std::sync::LazyLock;
use strum::IntoEnumIterator;

/// Decomposed training targets for a single move
/// Each field represents the target value for one policy head
#[derive(Debug, Clone)]
pub struct DecomposedTargets {
    pub action_type: usize, // 0-11: see move_type_to_idx (10 is EndTurn, 11 is Resign)
    pub source_spatial: Option<usize>, // which source tile (if applicable)
    pub target_spatial: Option<usize>, // which target tile (if applicable)
    pub target_type: Option<usize>, // 0-191: unified option head
}

impl Default for DecomposedTargets {
    fn default() -> Self {
        Self {
            action_type: 10,
            source_spatial: None,
            target_spatial: None,
            target_type: None,
        }
    }
}

// ============================================================================
// Robust Mapping Lookups
// ============================================================================

/// Width of the unified `move_option` head; `pi_option` in network.rs and
/// `NUM_OPTIONS` in train.py must agree.
pub const NUM_MOVE_OPTIONS: usize = 192;

// Offset constants for 192-sized head
pub const OFFSET_STRUCTURES: usize = 0;
pub const OFFSET_UNITS: usize = 48;
pub const OFFSET_TECHS: usize = 112;
pub const OFFSET_ABILITIES: usize = 160;
// Abilities only use 160..=180 (21 variants), so rewards fit in the same head.
pub const OFFSET_REWARDS: usize = 181;
// Legacy catch-all: pre-reward-slot data mapped every reward here.
pub const REWARD_FALLBACK_SLOT: usize = 191;

// Block capacities, derived from the offsets so the two cannot drift apart.
const MAX_STRUCTURES: usize = OFFSET_UNITS - OFFSET_STRUCTURES;
const MAX_UNITS: usize = OFFSET_TECHS - OFFSET_UNITS;
const MAX_TECHS: usize = OFFSET_ABILITIES - OFFSET_TECHS;
const MAX_ABILITIES: usize = OFFSET_REWARDS - OFFSET_ABILITIES;
const MAX_REWARDS: usize = REWARD_FALLBACK_SLOT - OFFSET_REWARDS;

/// Compile-time count of `$ty` variants outside `$skip`, by walking the
/// `#[repr(i8)]` space through strum's const `from_repr`.
macro_rules! repr_variant_count {
    ($ty:ty, $skip:pat) => {{
        let mut n = 0usize;
        let mut repr = i8::MIN;
        loop {
            match <$ty>::from_repr(repr) {
                Some($skip) | None => {}
                Some(_) => n += 1,
            }
            if repr == i8::MAX {
                break;
            }
            repr += 1;
        }
        n
    }};
}

const UNIT_SLOTS: usize = repr_variant_count!(UnitType, UnitType::None);
const TECH_SLOTS: usize = repr_variant_count!(
    TechnologyType,
    TechnologyType::Basic | TechnologyType::BeyondComprehension
);
// The ability block is full: 21 variants in 21 slots, with no free slot on
// either side of it. `ability_slot` is exhaustive, so a new AbilityType fails
// to build there rather than silently aliasing onto the reward block — but
// that only forces you to *write* an arm, and writing `=> 21` would compile
// and land the new ability on `OFFSET_REWARDS`. The const block below checks
// the mapping itself rather than this count, so that case fails to build too.
// Growing the block at all means re-laying-out the option head, which
// invalidates every trained `move_option` slot; treat it as a checkpoint
// migration, not an edit.
const ABILITY_SLOTS: usize = 21;
const REWARD_SLOTS: usize = 8;

// Every ability lands inside the ability block, and no two share a slot. This
// checks `ability_slot` itself; `ABILITY_SLOTS` below is a hardcoded count and
// asserting on it says nothing about a newly added variant.
const _: () = {
    let mut seen = [false; MAX_ABILITIES];
    let mut repr = i8::MIN;
    loop {
        if let Some(a) = AbilityType::from_repr(repr) {
            if let Some(slot) = ability_slot(a) {
                assert!(slot < MAX_ABILITIES, "ability slot escapes the ability block");
                assert!(!seen[slot], "two AbilityTypes share an option slot");
                seen[slot] = true;
            }
        }
        if repr == i8::MAX {
            break;
        }
        repr += 1;
    }
};

// Every family must fit between its own offset and the next one.
const _: () = {
    assert!(UNIT_SLOTS <= MAX_UNITS);
    assert!(TECH_SLOTS <= MAX_TECHS);
    assert!(ABILITY_SLOTS <= MAX_ABILITIES);
    assert!(REWARD_SLOTS <= MAX_REWARDS);
    assert!(REWARD_FALLBACK_SLOT < NUM_MOVE_OPTIONS);
};

// Producer/consumer width: every MoveType must land in a distinct slot of the
// action-type head that the writers and network.rs size with NUM_ACTION_TYPES.
const _: () = {
    let mut seen = [false; NUM_ACTION_TYPES];
    let mut repr = i8::MIN;
    loop {
        if let Some(mt) = MoveType::from_repr(repr) {
            let idx = DecomposedMapper::move_type_to_idx(mt);
            assert!(idx < NUM_ACTION_TYPES, "action_type head is too narrow");
            assert!(!seen[idx], "two MoveTypes share an action_type slot");
            seen[idx] = true;
        }
        if repr == i8::MAX {
            break;
        }
        repr += 1;
    }
};

static STRUCTURE_MAP: LazyLock<HashMap<StructureType, usize>> = LazyLock::new(|| {
    StructureType::iter()
        .filter(|&s| s != StructureType::None)
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect()
});

static UNIT_MAP: LazyLock<HashMap<UnitType, usize>> = LazyLock::new(|| {
    UnitType::iter()
        .filter(|&u| u != UnitType::None)
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect()
});

static TECH_MAP: LazyLock<HashMap<TechnologyType, usize>> = LazyLock::new(|| {
    TechnologyType::iter()
        .filter(|&t| t != TechnologyType::Basic && t != TechnologyType::BeyondComprehension)
        .enumerate()
        .map(|(i, t)| (t, i))
        .collect()
});

/// Slot inside the ability block, in `AbilityType` declaration order (what
/// `AbilityType::iter()` yields, so trained option slots keep their meaning).
/// Exhaustive on purpose: a new variant fails to compile here.
const fn ability_slot(a: AbilityType) -> Option<usize> {
    Some(match a {
        AbilityType::None => return None,
        AbilityType::BurnForest => 0,
        AbilityType::ClearForest => 1,
        AbilityType::GrowForest => 2,
        AbilityType::Destroy => 3,
        AbilityType::Decompose => 4,
        AbilityType::Convert => 5,
        AbilityType::Recover => 6,
        AbilityType::Disband => 7,
        AbilityType::HealOthers => 8,
        AbilityType::Drain => 9,
        AbilityType::FreezeArea => 10,
        AbilityType::Swarm => 11,
        AbilityType::Explode => 12,
        AbilityType::Promote => 13,
        AbilityType::BreakIce => 14,
        AbilityType::BreakPeace => 15,
        AbilityType::PeaceRequestResponse => 16,
        AbilityType::EstablishEmbassy => 17,
        AbilityType::PeaceTreaty => 18,
        AbilityType::DestroyEmbassy => 19,
        AbilityType::EnchantAnimal => 20,
    })
}

pub struct DecomposedMapper;

impl DecomposedMapper {
    pub const fn move_type_to_idx(mt: MoveType) -> usize {
        match mt {
            MoveType::None => 0,
            MoveType::Attack => 1,
            MoveType::Step => 2,
            MoveType::Capture => 3,
            MoveType::Ability => 4,
            MoveType::Summon => 5,
            MoveType::Harvest => 6,
            MoveType::Build => 7,
            MoveType::Research => 8,
            MoveType::Reward => 9,
            MoveType::EndTurn => 10,
            MoveType::Resign => 11,
        }
    }

    /// Convert a move into training targets for each policy head
    pub fn move_to_targets(m: &dyn Move, map_size: usize) -> DecomposedTargets {
        let move_type = m.move_type();
        let move_index = Self::move_type_to_idx(move_type);

        let source_spatial = m
            .source_idx()
            .ok()
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let target_spatial = m
            .target_idx()
            .ok()
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let target_type = match move_type {
            MoveType::Build => m.structure_type().ok().and_then(|s| Self::map_structure(s)),
            MoveType::Summon => m.unit_type().ok().and_then(|u| Self::map_unit(u)),
            MoveType::Research => m.tech_type().ok().and_then(|t| Self::map_tech(t)),
            MoveType::Ability => m.ability_type().ok().and_then(|a| Self::map_ability(a)),
            MoveType::Reward => Some(
                m.reward_type()
                    .ok()
                    .and_then(Self::map_reward)
                    .unwrap_or(REWARD_FALLBACK_SLOT),
            ),
            // Harvesting / Capturing does not require a target type
            _ => None,
        };

        DecomposedTargets {
            action_type: move_index,
            source_spatial,
            target_spatial,
            target_type,
        }
    }

    pub fn move_visit_to_targets(
        mv: &crate::ai::mcts_types::MoveVisit,
        map_size: usize,
    ) -> DecomposedTargets {
        let action_type = Self::move_type_to_idx(mv.move_type);

        let source_spatial = mv
            .source_idx
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));
        let target_spatial = mv
            .target_idx
            .map(|idx| Self::tile_to_spatial_idx(idx, map_size));

        let move_option = match mv.move_type {
            MoveType::Build => mv.structure_type.and_then(|s| Self::map_structure(s)),
            MoveType::Summon => mv.unit_type.and_then(|u| Self::map_unit(u)),
            MoveType::Research => mv.tech_type.and_then(|t| Self::map_tech(t)),
            MoveType::Ability => mv.ability_type.and_then(|a| Self::map_ability(a)),
            MoveType::Reward => Some(
                mv.reward_type
                    .and_then(Self::map_reward)
                    .unwrap_or(REWARD_FALLBACK_SLOT),
            ),
            _ => None,
        };

        DecomposedTargets {
            action_type,
            source_spatial,
            target_spatial,
            target_type: move_option,
        }
    }

    #[inline]
    fn tile_to_spatial_idx(tile_idx: usize, map_size: usize) -> usize {
        let y = tile_idx / map_size;
        let x = tile_idx % map_size;
        y * map_size + x
    }

    pub fn map_structure(s: StructureType) -> Option<usize> {
        if s == StructureType::None {
            return None;
        }
        STRUCTURE_MAP.get(&s).map(|&i| {
            if i >= MAX_STRUCTURES {
                eprintln!(
                    "[Mapper Warning] Structure {:?} index {} exceeds MAX_STRUCTURES {}",
                    s, i, MAX_STRUCTURES
                );
            }
            OFFSET_STRUCTURES + i
        })
    }

    pub fn map_unit(u: UnitType) -> Option<usize> {
        if u == UnitType::None {
            return None;
        }
        UNIT_MAP.get(&u).map(|&i| OFFSET_UNITS + i)
    }

    pub fn map_tech(t: TechnologyType) -> Option<usize> {
        if t == TechnologyType::Basic || t == TechnologyType::BeyondComprehension {
            return None;
        }
        TECH_MAP.get(&t).map(|&i| {
            if i >= MAX_TECHS {
                eprintln!(
                    "[Mapper Warning] Tech {:?} index {} exceeds MAX_TECHS {}",
                    t, i, MAX_TECHS
                );
            }
            OFFSET_TECHS + i
        })
    }

    /// Explicit slots (not enum-iteration order) so the mapping stays stable
    /// even if variants are reordered. Each city-reward choice gets its own
    /// option-head slot — before this, every reward shared slot 191 and the
    /// policy could not express Workshop-vs-Explorer etc. at all.
    pub const fn map_reward(r: CityRewardType) -> Option<usize> {
        let i = match r {
            CityRewardType::None => return None,
            CityRewardType::CityWall => 0,
            CityRewardType::Park => 1,
            CityRewardType::Workshop => 2,
            CityRewardType::Explorer => 3,
            CityRewardType::BorderGrowth => 4,
            CityRewardType::SuperUnit => 5,
            CityRewardType::Resources => 6,
            CityRewardType::PopGrowth => 7,
        };
        Some(OFFSET_REWARDS + i)
    }

    pub const fn map_ability(a: AbilityType) -> Option<usize> {
        match ability_slot(a) {
            Some(i) => Some(OFFSET_ABILITIES + i),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::features::MAP_SIZE;
    use crate::ai::mcts_types::MoveVisit;
    use crate::moves::{
        AttackMove, BuildMove, CaptureMove, DestroyMove, EndTurnMove, HarvestMove, ResearchMove,
        ResignMove, RewardMove, StepMove, SummonMove,
    };
    use std::collections::HashSet;

    fn visit(move_type: MoveType) -> MoveVisit {
        MoveVisit {
            move_type,
            visits: 1.0,
            source_idx: None,
            target_idx: None,
            structure_type: None,
            unit_type: None,
            tech_type: None,
            ability_type: None,
            reward_type: None,
        }
    }

    #[test]
    fn move_types_fill_the_action_head() {
        let mut seen = HashSet::new();
        let mut variants = 0;
        for repr in i8::MIN..=i8::MAX {
            let Some(mt) = MoveType::from_repr(repr) else {
                continue;
            };
            let idx = DecomposedMapper::move_type_to_idx(mt);
            assert!(idx < NUM_ACTION_TYPES, "{mt:?} maps to {idx}");
            assert!(seen.insert(idx), "{mt:?} collides on action slot {idx}");
            variants += 1;
        }
        assert_eq!(
            variants, NUM_ACTION_TYPES,
            "action head width != MoveType count"
        );
    }

    /// Trained checkpoints depend on these exact slots.
    #[test]
    fn action_type_slots_are_stable() {
        let expected = [
            (MoveType::None, 0),
            (MoveType::Attack, 1),
            (MoveType::Step, 2),
            (MoveType::Capture, 3),
            (MoveType::Ability, 4),
            (MoveType::Summon, 5),
            (MoveType::Harvest, 6),
            (MoveType::Build, 7),
            (MoveType::Research, 8),
            (MoveType::Reward, 9),
            (MoveType::EndTurn, 10),
            (MoveType::Resign, 11),
        ];
        for (mt, idx) in expected {
            assert_eq!(DecomposedMapper::move_type_to_idx(mt), idx, "{mt:?}");
        }
    }

    #[test]
    fn option_blocks_have_room_for_their_enums() {
        let structures = StructureType::iter()
            .filter(|&s| s != StructureType::None)
            .count();
        let units = UnitType::iter().filter(|&u| u != UnitType::None).count();
        let techs = TechnologyType::iter()
            .filter(|&t| t != TechnologyType::Basic && t != TechnologyType::BeyondComprehension)
            .count();
        let abilities = AbilityType::iter()
            .filter(|&a| a != AbilityType::None)
            .count();
        let rewards = CityRewardType::iter()
            .filter(|&r| r != CityRewardType::None)
            .count();

        assert!(structures <= MAX_STRUCTURES, "{structures} structures");
        assert_eq!(units, UNIT_SLOTS);
        assert_eq!(techs, TECH_SLOTS);
        assert_eq!(
            abilities, ABILITY_SLOTS,
            "ability block is full at {MAX_ABILITIES}"
        );
        assert_eq!(rewards, REWARD_SLOTS);
    }

    #[test]
    fn option_blocks_do_not_overlap() {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut claim = |slot: Option<usize>, lo: usize, hi: usize, what: String| {
            let slot = slot.unwrap_or_else(|| panic!("{what} has no option slot"));
            assert!(
                slot >= lo && slot < hi,
                "{what} -> {slot}, outside {lo}..{hi}"
            );
            assert!(seen.insert(slot), "{what} collides on option slot {slot}");
        };

        for s in StructureType::iter().filter(|&s| s != StructureType::None) {
            claim(
                DecomposedMapper::map_structure(s),
                OFFSET_STRUCTURES,
                OFFSET_UNITS,
                format!("{s:?}"),
            );
        }
        for u in UnitType::iter().filter(|&u| u != UnitType::None) {
            claim(
                DecomposedMapper::map_unit(u),
                OFFSET_UNITS,
                OFFSET_TECHS,
                format!("{u:?}"),
            );
        }
        for t in TechnologyType::iter()
            .filter(|&t| t != TechnologyType::Basic && t != TechnologyType::BeyondComprehension)
        {
            claim(
                DecomposedMapper::map_tech(t),
                OFFSET_TECHS,
                OFFSET_ABILITIES,
                format!("{t:?}"),
            );
        }
        for a in AbilityType::iter().filter(|&a| a != AbilityType::None) {
            claim(
                DecomposedMapper::map_ability(a),
                OFFSET_ABILITIES,
                OFFSET_REWARDS,
                format!("{a:?}"),
            );
        }
        for r in CityRewardType::iter().filter(|&r| r != CityRewardType::None) {
            claim(
                DecomposedMapper::map_reward(r),
                OFFSET_REWARDS,
                REWARD_FALLBACK_SLOT,
                format!("{r:?}"),
            );
        }

        assert!(!seen.contains(&REWARD_FALLBACK_SLOT));
        assert!(seen.iter().all(|&s| s < NUM_MOVE_OPTIONS));
    }

    /// The explicit ability slots must reproduce the enum-iteration order the
    /// previous lookup table used, or trained option slots shift meaning.
    #[test]
    fn ability_slots_follow_enum_iteration_order() {
        assert_eq!(DecomposedMapper::map_ability(AbilityType::None), None);
        for (i, a) in AbilityType::iter()
            .filter(|&a| a != AbilityType::None)
            .enumerate()
        {
            assert_eq!(
                DecomposedMapper::map_ability(a),
                Some(OFFSET_ABILITIES + i),
                "{a:?}"
            );
        }
    }

    #[test]
    fn representative_moves_round_trip() {
        let idx_of = DecomposedMapper::move_type_to_idx;

        let t = DecomposedMapper::move_to_targets(&StepMove::new(5, 17), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Step));
        assert_eq!(t.source_spatial, Some(5));
        assert_eq!(t.target_spatial, Some(17));
        assert_eq!(t.target_type, None);

        let t = DecomposedMapper::move_to_targets(&AttackMove::new(5, 6), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Attack));
        assert_eq!((t.source_spatial, t.target_spatial), (Some(5), Some(6)));

        let t = DecomposedMapper::move_to_targets(&CaptureMove::new(42), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Capture));
        assert_eq!((t.source_spatial, t.target_spatial), (Some(42), None));

        let t = DecomposedMapper::move_to_targets(&HarvestMove::new(7), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Harvest));
        assert_eq!((t.source_spatial, t.target_spatial), (None, Some(7)));

        let t =
            DecomposedMapper::move_to_targets(&BuildMove::new(9, StructureType::Farm), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Build));
        assert_eq!(t.target_spatial, Some(9));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_structure(StructureType::Farm)
        );

        let t = DecomposedMapper::move_to_targets(&SummonMove::new(3, UnitType::Rider), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Summon));
        assert_eq!(t.source_spatial, Some(3));
        assert_eq!(t.target_type, DecomposedMapper::map_unit(UnitType::Rider));

        let t =
            DecomposedMapper::move_to_targets(&ResearchMove::new(TechnologyType::Riding), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Research));
        assert_eq!((t.source_spatial, t.target_spatial), (None, None));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_tech(TechnologyType::Riding)
        );

        let t = DecomposedMapper::move_to_targets(&DestroyMove::new(11), MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Ability));
        assert_eq!(t.target_spatial, Some(11));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_ability(AbilityType::Destroy)
        );

        let t = DecomposedMapper::move_to_targets(
            &RewardMove::new(4, CityRewardType::Workshop),
            MAP_SIZE,
        );
        assert_eq!(t.action_type, idx_of(MoveType::Reward));
        assert_eq!(t.target_spatial, Some(4));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_reward(CityRewardType::Workshop)
        );

        let t = DecomposedMapper::move_to_targets(&EndTurnMove, MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::EndTurn));
        assert_eq!(
            (t.source_spatial, t.target_spatial, t.target_type),
            (None, None, None)
        );

        let t = DecomposedMapper::move_to_targets(&ResignMove, MAP_SIZE);
        assert_eq!(t.action_type, idx_of(MoveType::Resign));
    }

    #[test]
    fn move_visits_map_like_moves() {
        let mut mv = visit(MoveType::Build);
        mv.target_idx = Some(9);
        mv.structure_type = Some(StructureType::Farm);
        let t = DecomposedMapper::move_visit_to_targets(&mv, MAP_SIZE);
        assert_eq!(
            t.action_type,
            DecomposedMapper::move_type_to_idx(MoveType::Build)
        );
        assert_eq!(t.target_spatial, Some(9));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_structure(StructureType::Farm)
        );

        let mut mv = visit(MoveType::Ability);
        mv.source_idx = Some(2);
        mv.ability_type = Some(AbilityType::Convert);
        let t = DecomposedMapper::move_visit_to_targets(&mv, MAP_SIZE);
        assert_eq!(t.source_spatial, Some(2));
        assert_eq!(
            t.target_type,
            DecomposedMapper::map_ability(AbilityType::Convert)
        );

        // A reward visit with no reward recorded keeps the legacy catch-all slot.
        let t = DecomposedMapper::move_visit_to_targets(&visit(MoveType::Reward), MAP_SIZE);
        assert_eq!(t.target_type, Some(REWARD_FALLBACK_SLOT));
    }
}
