// Implement Move trait helpers for Step

use super::Move;
use crate::types::*;

impl crate::moves::StepMove {
    #[inline]
    pub fn source_idx_impl(&self) -> Option<usize> {
        Some(self.from_idx as usize)
    }

    #[inline]
    pub fn target_idx_impl(&self) -> Option<usize> {
        Some(self.to_idx as usize)
    }
}
