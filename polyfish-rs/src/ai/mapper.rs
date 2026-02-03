use crate::moves::Move;
use crate::types::MoveType;

pub struct ActionMapper;

impl ActionMapper {
    pub const TOTAL_CHANNELS: usize = 64;
    pub const TOTAL_ACTIONS: usize = 30 * 30 * 64;

    #[inline]
    pub fn move_to_idx(game_size: i32, m: &dyn Move) -> Option<usize> {
        let mt = m.move_type();

        // Use action_coords() instead of serialize() for performance
        let (src_opt, target_opt) = m.action_coords();

        // Panic if action_coords not implemented for moves that need tile info
        // EndTurn and Research don't need coordinates
        let src = match mt {
            MoveType::EndTurn => 0,
            MoveType::Research => 0,
            _ => src_opt.unwrap_or_else(|| {
                panic!(
                    "action_coords() not implemented for {:?}: {:?}",
                    mt,
                    m.describe(&crate::states::GameState::default())
                )
            }),
        };
        let target = target_opt.unwrap_or(src);

        let w = game_size;
        let (sx, sy) = (src % w, src / w);
        let (tx, ty) = (target % w, target / w);
        let dx = tx - sx;
        let dy = ty - sy;

        let channel = match mt {
            MoveType::Step => direction_channel(dx, dy, 0),
            MoveType::Attack => attack_channel(dx, dy),
            MoveType::Capture => Some(32),
            MoveType::Ability => Some(33),
            MoveType::Build => Some(35),
            MoveType::EndTurn => Some(63),
            _ => Some(62), // Others (Research, Reward, Summon, etc.)
        }?;

        let plane_size = (30 * 30) as usize;
        let pixel_idx = (sy * 30 + sx) as usize;

        if pixel_idx >= 900 {
            return None;
        }

        Some(channel * plane_size + pixel_idx)
    }
}

#[inline]
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

#[inline]
fn attack_channel(dx: i32, dy: i32) -> Option<usize> {
    if let Some(c) = direction_channel(dx, dy, 8) {
        return Some(c);
    }
    Some(31)
}
