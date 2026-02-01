use crate::moves::Move;
use crate::types::MoveType;

pub struct ActionMapper;

impl ActionMapper {
    pub const TOTAL_CHANNELS: usize = 64;
    pub const TOTAL_ACTIONS: usize = 30 * 30 * 64;

    pub fn move_to_idx(game_size: i32, m: &dyn Move) -> Option<usize> {
        let mt = m.move_type();
        let json = m.serialize();

        // Robustly get src/target from various move types
        // Step/Attack/Ability usually have src/target or tile_index
        let src = if let Some(val) = json.get("src") {
            val.as_i64().unwrap_or(0) as i32
        } else if let Some(val) = json.get("tile_index") {
            // For build/recover, often only one tile matters
            val.as_i64().unwrap_or(0) as i32
        } else {
            0
        };

        let target = if let Some(val) = json.get("target") {
            val.as_i64().unwrap_or(src as i64) as i32
        } else {
            src // Default to self if no target
        };

        let w = game_size;
        let (sx, sy) = (src % w, src / w);
        let (tx, ty) = (target % w, target / w);
        let dx = tx - sx;
        let dy = ty - sy;

        let channel = match mt {
            MoveType::Step => direction_channel(dx, dy, 0),
            MoveType::Attack => attack_channel(dx, dy),
            MoveType::Capture => Some(32),
            MoveType::Ability => {
                // Check ability type? "abilityType" in json
                // For now bucket all abilities
                Some(33)
            }
            MoveType::Build => Some(35),
            MoveType::EndTurn => Some(63),
            _ => Some(62), // Others
        }?;

        let plane_size = (30 * 30) as usize;
        // Map (sx, sy) to flat index within the plane
        // 30x30 max size assumption for network
        let pixel_idx = (sy * 30 + sx) as usize;

        if pixel_idx >= 900 {
            return None;
        }

        Some(channel * plane_size + pixel_idx)
    }
}

fn direction_channel(dx: i32, dy: i32, base: usize) -> Option<usize> {
    match (dx, dy) {
        (0, -1) => Some(base + 0),
        (1, -1) => Some(base + 1),
        (1, 0) => Some(base + 2),
        (1, 1) => Some(base + 3),
        (0, 1) => Some(base + 4),
        (-1, 1) => Some(base + 5),
        (-1, 0) => Some(base + 6),
        (-1, -1) => Some(base + 7),
        _ => None,
    }
}

fn attack_channel(dx: i32, dy: i32) -> Option<usize> {
    if let Some(c) = direction_channel(dx, dy, 8) {
        return Some(c);
    }
    Some(31)
}
