use std::ops::{Index, IndexMut};

use indexmap::{map, IndexMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::TileState;

/// Stores complete row-major boards directly and preserves insertion order for
/// older or custom boards whose keys are sparse or out of order.
#[derive(Clone, Debug)]
pub struct BoardTiles(Storage);

#[derive(Clone, Debug)]
enum Storage {
    Indexed(Vec<(i32, TileState)>),
    Ordered(IndexMap<i32, TileState>),
}

impl BoardTiles {
    pub fn get(&self, key: &i32) -> Option<&TileState> {
        match &self.0 {
            Storage::Indexed(tiles) => tiles.get(*key as usize).map(|(_, tile)| tile),
            Storage::Ordered(tiles) => tiles.get(key),
        }
    }

    pub fn get_mut(&mut self, key: &i32) -> Option<&mut TileState> {
        match &mut self.0 {
            Storage::Indexed(tiles) => tiles.get_mut(*key as usize).map(|(_, tile)| tile),
            Storage::Ordered(tiles) => tiles.get_mut(key),
        }
    }

    pub fn get_full(&self, key: &i32) -> Option<(usize, &i32, &TileState)> {
        match &self.0 {
            Storage::Indexed(tiles) => {
                let position = *key as usize;
                tiles.get(position).map(|(idx, tile)| (position, idx, tile))
            }
            Storage::Ordered(tiles) => tiles.get_full(key),
        }
    }

    pub fn contains_key(&self, key: &i32) -> bool {
        self.get(key).is_some()
    }

    pub fn insert(&mut self, key: i32, tile: TileState) -> Option<TileState> {
        match &mut self.0 {
            Storage::Ordered(tiles) => return tiles.insert(key, tile),
            Storage::Indexed(tiles) => {
                if key as usize == tiles.len() {
                    tiles.push((key, tile));
                    return None;
                }
                if let Some((_, previous)) = tiles.get_mut(key as usize) {
                    return Some(std::mem::replace(previous, tile));
                }
                let mut ordered: IndexMap<_, _> = std::mem::take(tiles).into_iter().collect();
                ordered.insert(key, tile);
                self.0 = Storage::Ordered(ordered);
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        match &self.0 {
            Storage::Indexed(tiles) => tiles.len(),
            Storage::Ordered(tiles) => tiles.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> BoardTilesIter<'_> {
        match &self.0 {
            Storage::Indexed(tiles) => BoardTilesIter::Indexed(tiles.iter()),
            Storage::Ordered(tiles) => BoardTilesIter::Ordered(tiles.iter()),
        }
    }

    pub fn iter_mut(&mut self) -> BoardTilesIterMut<'_> {
        match &mut self.0 {
            Storage::Indexed(tiles) => BoardTilesIterMut::Indexed(tiles.iter_mut()),
            Storage::Ordered(tiles) => BoardTilesIterMut::Ordered(tiles.iter_mut()),
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &i32> {
        self.iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl Iterator<Item = &TileState> {
        self.iter().map(|(_, tile)| tile)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut TileState> {
        self.iter_mut().map(|(_, tile)| tile)
    }
}

impl Default for BoardTiles {
    fn default() -> Self {
        Self(Storage::Indexed(Vec::new()))
    }
}

impl From<IndexMap<i32, TileState>> for BoardTiles {
    fn from(tiles: IndexMap<i32, TileState>) -> Self {
        if tiles
            .keys()
            .enumerate()
            .all(|(position, &key)| key as usize == position)
        {
            Self(Storage::Indexed(tiles.into_iter().collect()))
        } else {
            Self(Storage::Ordered(tiles))
        }
    }
}

impl Index<&i32> for BoardTiles {
    type Output = TileState;

    fn index(&self, key: &i32) -> &Self::Output {
        self.get(key).expect("Tile index not found")
    }
}

impl IndexMut<&i32> for BoardTiles {
    fn index_mut(&mut self, key: &i32) -> &mut Self::Output {
        self.get_mut(key).expect("Tile index not found")
    }
}

pub enum BoardTilesIter<'a> {
    Indexed(std::slice::Iter<'a, (i32, TileState)>),
    Ordered(map::Iter<'a, i32, TileState>),
}

impl<'a> Iterator for BoardTilesIter<'a> {
    type Item = (&'a i32, &'a TileState);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed(iter) => iter.next().map(|(key, tile)| (key, tile)),
            Self::Ordered(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Indexed(iter) => iter.size_hint(),
            Self::Ordered(iter) => iter.size_hint(),
        }
    }
}

impl<'a> IntoIterator for &'a BoardTiles {
    type Item = (&'a i32, &'a TileState);
    type IntoIter = BoardTilesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub enum BoardTilesIterMut<'a> {
    Indexed(std::slice::IterMut<'a, (i32, TileState)>),
    Ordered(map::IterMut<'a, i32, TileState>),
}

impl<'a> Iterator for BoardTilesIterMut<'a> {
    type Item = (&'a i32, &'a mut TileState);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Indexed(iter) => iter.next().map(|(key, tile)| (&*key, tile)),
            Self::Ordered(iter) => iter.next(),
        }
    }
}

impl<'a> IntoIterator for &'a mut BoardTiles {
    type Item = (&'a i32, &'a mut TileState);
    type IntoIter = BoardTilesIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl Serialize for BoardTiles {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

impl<'de> Deserialize<'de> for BoardTiles {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        IndexMap::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_board_uses_keys_independent_of_coordinates() {
        let mut tiles = BoardTiles::default();
        tiles.insert(0, TileState::default());
        tiles.insert(1, TileState::default());
        tiles[&1].has_road = true;
        assert_eq!(tiles.get_full(&1).unwrap().0, 1);
        assert!(tiles[&1].has_road);
        assert!(tiles.get(&-1).is_none());
        assert!(matches!(tiles.0, Storage::Indexed(_)));
    }

    #[test]
    fn sparse_insert_and_json_preserve_insertion_order() {
        let mut tiles = BoardTiles::default();
        for idx in [0, 10, 1, -1] {
            tiles.insert(idx, TileState::default());
        }
        let original: Vec<_> = tiles.keys().copied().collect();
        assert_eq!(original, [0, 10, 1, -1]);
        let replaced = tiles.insert(
            10,
            TileState {
                has_road: true,
                ..Default::default()
            },
        );
        assert!(replaced.is_some());
        assert_eq!(tiles.get_full(&10).unwrap().0, 1);
        let json = serde_json::to_string(&tiles).unwrap();
        assert!(json.starts_with('{'));
        let restored: BoardTiles = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.keys().copied().collect::<Vec<_>>(), original);
        assert!(restored[&10].has_road);
        assert!(matches!(restored.0, Storage::Ordered(_)));
    }
}
