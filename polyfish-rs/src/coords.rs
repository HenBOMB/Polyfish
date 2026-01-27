//! Coordinate utilities

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Represents a position on the game map with x, y and flat index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Coords {
    pub x: i32,
    pub y: i32,
    pub idx: i32,
}

// Custom serialization: serialize as [x, y] array to match TypeScript format
impl Serialize for Coords {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as [x, y] array
        (self.x, self.y).serialize(serializer)
    }
}

// Custom deserialization: deserialize from [x, y] array
impl<'de> Deserialize<'de> for Coords {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize as [x, y] tuple
        let (x, y) = <(i32, i32)>::deserialize(deserializer)?;
        // Note: idx will be computed later when we know the map size
        // For now, compute assuming a reasonable default or leave as placeholder
        Ok(Coords { x, y, idx: -1 })
    }
}

impl Coords {
    /// Create new coords from index and map size
    pub fn from_index(index: i32, map_size: i32) -> Self {
        Self {
            idx: index,
            x: index % map_size,
            y: index / map_size,
        }
    }

    /// Create new coords from x, y and map size
    pub fn from_xy(x: i32, y: i32, map_size: i32) -> Self {
        Self {
            x,
            y,
            idx: y * map_size + x,
        }
    }

    /// Create invalid/unset coords
    pub fn invalid() -> Self {
        Self { x: -1, y: -1, idx: -1 }
    }

    /// Check if coords are valid
    pub fn is_valid(&self) -> bool {
        self.x >= 0 && self.y >= 0
    }

    /// Compute and set the idx based on map size
    pub fn compute_idx(&mut self, map_size: i32) {
        self.idx = self.y * map_size + self.x;
    }

    /// Set from index
    pub fn set_at(&mut self, index: i32, map_size: i32) {
        self.idx = index;
        self.x = index % map_size;
        self.y = index / map_size;
    }

    /// Set from x, y
    pub fn set(&mut self, x: i32, y: i32, map_size: i32) {
        self.x = x;
        self.y = y;
        self.idx = y * map_size + x;
    }

    /// Copy from another coords
    pub fn copy_from(&mut self, other: &Coords) {
        self.x = other.x;
        self.y = other.y;
        self.idx = other.idx;
    }

    /// Calculate manhattan distance to another coords
    pub fn distance_to(&self, other: &Coords) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }

    /// Calculate chebyshev distance (for diagonal movement)
    pub fn chebyshev_distance_to(&self, other: &Coords) -> i32 {
        (self.x - other.x).abs().max((self.y - other.y).abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_index() {
        let coords = Coords::from_index(15, 11);
        assert_eq!(coords.x, 4);
        assert_eq!(coords.y, 1);
        assert_eq!(coords.idx, 15);
    }

    #[test]
    fn test_from_xy() {
        let coords = Coords::from_xy(4, 1, 11);
        assert_eq!(coords.idx, 15);
    }

    #[test]
    fn test_distance() {
        let a = Coords::from_xy(0, 0, 11);
        let b = Coords::from_xy(3, 4, 11);
        assert_eq!(a.distance_to(&b), 7);
        assert_eq!(a.chebyshev_distance_to(&b), 4);
    }
    
    #[test]
    fn test_serde_roundtrip() {
        let coords = Coords::from_xy(5, 3, 11);
        let json = serde_json::to_string(&coords).unwrap();
        assert_eq!(json, "[5,3]");
        
        let parsed: Coords = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.x, 5);
        assert_eq!(parsed.y, 3);
    }
}
