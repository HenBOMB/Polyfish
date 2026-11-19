use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use strum::IntoEnumIterator;

use crate::{states::PlayerId, types::TileEffect};

/// Player visibility stored inline for normal seat IDs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileExplorers {
    mask: u64,
    overflow: Option<Box<BTreeSet<PlayerId>>>,
}

impl TileExplorers {
    pub fn contains(&self, player: &PlayerId) -> bool {
        match Self::bit(*player) {
            Some(bit) => self.mask & bit != 0,
            None => self
                .overflow
                .as_ref()
                .is_some_and(|ids| ids.contains(player)),
        }
    }

    pub fn insert(&mut self, player: PlayerId) -> bool {
        let Some(bit) = Self::bit(player) else {
            return self
                .overflow
                .get_or_insert_with(Box::default)
                .insert(player);
        };
        let inserted = self.mask & bit == 0;
        self.mask |= bit;
        inserted
    }

    pub fn remove(&mut self, player: &PlayerId) -> bool {
        if let Some(bit) = Self::bit(*player) {
            let removed = self.mask & bit != 0;
            self.mask &= !bit;
            return removed;
        }
        let Some(overflow) = &mut self.overflow else {
            return false;
        };
        let removed = overflow.remove(player);
        if overflow.is_empty() {
            self.overflow = None;
        }
        removed
    }

    pub fn clear(&mut self) {
        self.mask = 0;
        self.overflow = None;
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&PlayerId) -> bool) {
        for id in Self::inline_ids(self.mask) {
            if !keep(&id) {
                self.mask &= !(1 << id);
            }
        }
        if let Some(overflow) = &mut self.overflow {
            overflow.retain(keep);
            if overflow.is_empty() {
                self.overflow = None;
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = PlayerId> + '_ {
        Self::inline_ids(self.mask).chain(self.overflow.iter().flat_map(|ids| ids.iter().copied()))
    }

    fn bit(id: PlayerId) -> Option<u64> {
        // The cast also sends negative IDs to overflow.
        1u64.checked_shl(id as u32)
    }

    fn inline_ids(mut mask: u64) -> impl Iterator<Item = PlayerId> {
        std::iter::from_fn(move || {
            if mask == 0 {
                return None;
            }
            let player = mask.trailing_zeros();
            mask &= mask - 1;
            Some(player as PlayerId)
        })
    }
}

impl Extend<PlayerId> for TileExplorers {
    fn extend<T: IntoIterator<Item = PlayerId>>(&mut self, ids: T) {
        for id in ids {
            self.insert(id);
        }
    }
}

impl FromIterator<PlayerId> for TileExplorers {
    fn from_iter<T: IntoIterator<Item = PlayerId>>(ids: T) -> Self {
        let mut explorers = Self::default();
        explorers.extend(ids);
        explorers
    }
}

impl Serialize for TileExplorers {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

impl<'de> Deserialize<'de> for TileExplorers {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Vec::<PlayerId>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

/// Membership bits for the small, fixed set of tile effects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileEffects(u8);

impl TileEffects {
    pub fn contains(&self, effect: &TileEffect) -> bool {
        self.0 & Self::bit(*effect) != 0
    }

    pub fn insert(&mut self, effect: TileEffect) -> bool {
        let inserted = !self.contains(&effect);
        self.0 |= Self::bit(effect);
        inserted
    }

    pub fn remove(&mut self, effect: &TileEffect) -> bool {
        let removed = self.contains(effect);
        self.0 &= !Self::bit(*effect);
        removed
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }

    fn bit(effect: TileEffect) -> u8 {
        1 << effect as u8
    }
}

impl Serialize for TileEffects {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(TileEffect::iter().filter(|effect| self.contains(effect)))
    }
}

impl<'de> Deserialize<'de> for TileEffects {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut effects = Self::default();
        for effect in Vec::deserialize(deserializer)? {
            effects.insert(effect);
        }
        Ok(effects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TileState;
    use serde_json::json;

    #[test]
    fn visibility_keeps_all_legacy_player_ids() {
        let mut explorers = TileExplorers::default();
        let ids = [i32::MIN, -1, 0, 1, 31, 32, 63, 64, i32::MAX];
        explorers.extend(ids);
        for id in ids {
            assert!(explorers.contains(&id));
            assert!(!explorers.insert(id));
        }
        let original = explorers.clone();
        explorers.retain(|&id| id == 63 || id == i32::MAX);
        assert_eq!(explorers.iter().collect::<Vec<_>>(), vec![63, i32::MAX]);
        assert!(explorers.remove(&i32::MAX));
        assert!(!explorers.remove(&i32::MAX));
        assert!(explorers.overflow.is_none());
        explorers.clear();
        assert_eq!(explorers.iter().count(), 0);
        assert_eq!(original.iter().count(), ids.len());
    }

    #[test]
    fn legacy_tile_arrays_round_trip_with_set_semantics() {
        assert_eq!(std::mem::size_of::<TileEffects>(), 1);
        for effect in TileEffect::iter() {
            assert!((effect as u8) < 8);
        }
        let legacy = json!({
            "coords": {"x": 1, "y": 1, "idx": 12},
            "type": 3,
            "explorers": [1, -2, 1, 64],
            "effects": [0, 4, 1, 4]
        });
        let tile: TileState = serde_json::from_value(legacy).unwrap();
        let saved = serde_json::to_value(&tile).unwrap();
        assert_eq!(saved["explorers"], json!([1, -2, 64]));
        assert_eq!(saved["effects"], json!([0, 1, 4]));

        let mut restored: TileState = serde_json::from_value(saved).unwrap();
        assert_eq!(tile.explorers, restored.explorers);
        assert_eq!(tile.effects, restored.effects);
        assert!(restored.effects.remove(&TileEffect::Algae));
        assert!(!restored.effects.remove(&TileEffect::Algae));
        restored.effects.clear();
        assert_eq!(serde_json::to_value(restored.effects).unwrap(), json!([]));
    }

    #[test]
    fn game_loads_legacy_wrapped_tiles_and_saves_same_membership() {
        let legacy = json!({
            "settings": crate::GameSettings::default(),
            "map": {
                "tiles": [{"idx": 12, "x": 1, "y": 1, "type": 3, "explorers": [63, 64, -7], "effects": [4]}],
                "structures": [], "resources": []
            }
        });
        let game = crate::game::Game::from_json(&legacy.to_string()).unwrap();
        let restored = crate::game::Game::from_json(&game.to_json().unwrap()).unwrap();
        let tile = &restored.state.tiles[&12];
        for player in [63, 64, -7] {
            assert!(tile.explorers.contains(&player));
        }
        assert!(tile.effects.contains(&TileEffect::Algae));
    }
}
