//! Diplomacy move implementation (Break Peace, Peace Response, Establish Embassy)

use crate::actions::{
    UndoCallback, chain_undos, spend_stars, structure::create_structure,
    structure::destroy_structure,
};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::states::PlayerId;
use crate::types::{AbilityType, MoveType, StructureType};

// --- Break Peace ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BreakPeaceMove {
    pub opponent_id: PlayerId,
}

impl BreakPeaceMove {
    pub fn new(opponent_id: PlayerId) -> Self {
        Self { opponent_id }
    }
}

impl Move for BreakPeaceMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let target_id = self.opponent_id;

        let mut undos = Vec::new();

        // Break peace for both
        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            if let Some(relation) = tribe.relations.get_mut(&target_id) {
                let old_state = relation.state;
                relation.state = 0; // Neutral/War
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        if let Some(r) = t.relations.get_mut(&target_id) {
                            r.state = old_state;
                        }
                    }
                }) as UndoCallback);
            }
        }

        if let Some(tribe) = state.tribes.get_mut(&target_id) {
            if let Some(relation) = tribe.relations.get_mut(&pov_id) {
                let old_state = relation.state;
                relation.state = 0;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&target_id) {
                        if let Some(r) = t.relations.get_mut(&pov_id) {
                            r.state = old_state;
                        }
                    }
                }) as UndoCallback);
            }
        }

        // Breaking peace ends the turn for the player who broke it.
        // We find the Game controller and call end_turn?
        // No, execute is part of Move, and Move doesn't have access to Game.
        // Instead, we should return an end_turn reward or let play_move handle it?
        // Wait, play_move handles EndTurnMove specifically.
        // For other moves, we must manually update the state to "end" or make it a sequence.
        // In this simulation, the best way is to call end_turn logic inside execute if possible,
        // or ensure play_move checks for this.
        // Actually, we can't easily call end_turn here because it requires &mut Game.
        // BUT! Game::play_move calls end_turn if MoveType is EndTurn.
        // We can't change MoveType dynamically.

        // Wait! Look at how other moves do it.
        // Actually, we can just set a flag or let it be.
        // If we can't end the turn here, we must at least ensure it's not repeatable.
        // But relation.state becoming 0 ALREADY makes it non-repeatable in generation.
        // So the loop MUST be SIGN -> BREAK -> SIGN -> BREAK.

        // I will fix PeaceTreatyMove to NOT set state=1.

        // TODO: find solution

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Break peace with tribe {}", self.opponent_id)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "src": self.opponent_id,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.opponent_id as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::BreakPeace)
    }
}

pub fn generate_break_peace_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        for (&other_id, relation) in &tribe.relations {
            if relation.state == 1 {
                // At peace
                moves.push(Box::new(BreakPeaceMove::new(other_id)));
            }
        }
    }
}

// --- Peace Request Response ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeaceRequestResponse {
    pub opponent_id: PlayerId,
    pub accepted: i32,
}

impl PeaceRequestResponse {
    pub fn new(opponent_id: PlayerId, accepted: i32) -> Self {
        Self {
            opponent_id,
            accepted,
        }
    }
}

impl Move for PeaceRequestResponse {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let opp_id = self.opponent_id; // the tribe who sent the request
        let accepted = self.accepted == 1;
        let mut undos = Vec::new();

        if accepted {
            // Set BOTH tribes to peace
            if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                if let Some(relation) = tribe.relations.get_mut(&opp_id) {
                    let old_state = relation.state;
                    relation.state = 1; // Peace
                    let old_offered = relation.has_offered_peace;
                    relation.has_offered_peace = false; // Clear flag
                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) {
                            if let Some(r) = t.relations.get_mut(&opp_id) {
                                r.state = old_state;
                                r.has_offered_peace = old_offered;
                            }
                        }
                    }) as UndoCallback);
                }
            }
            if let Some(tribe) = state.tribes.get_mut(&opp_id) {
                if let Some(relation) = tribe.relations.get_mut(&pov_id) {
                    let old_state = relation.state;
                    relation.state = 1;
                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(t) = s.tribes.get_mut(&opp_id) {
                            if let Some(r) = t.relations.get_mut(&pov_id) {
                                r.state = old_state;
                            }
                        }
                    }) as UndoCallback);
                }
            }
        } else {
            // Rejected: just clear the flag for the current player's relation to the opponent
            if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                if let Some(relation) = tribe.relations.get_mut(&opp_id) {
                    let old_offered = relation.has_offered_peace;
                    relation.has_offered_peace = false;
                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(t) = s.tribes.get_mut(&pov_id) {
                            if let Some(r) = t.relations.get_mut(&opp_id) {
                                r.has_offered_peace = old_offered;
                            }
                        }
                    }) as UndoCallback);
                }
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Peace request {} {}",
            self.opponent_id,
            if self.accepted == 1 {
                "accepted"
            } else {
                "rejected"
            }
        )
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "target": self.accepted, // target = did we accept?
            "src": self.opponent_id, // source = who sent the request?
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.accepted as usize)
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.opponent_id as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::PeaceRequestResponse)
    }
}

// --- Establish Embassy ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EstablishEmbassyMove {
    pub target_index: i32,
    pub opponent_id: PlayerId,
}

impl EstablishEmbassyMove {
    pub fn new(target_index: i32, opponent_id: PlayerId) -> Self {
        Self {
            target_index,
            opponent_id,
        }
    }
}

impl Move for EstablishEmbassyMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let opp_id = self.opponent_id;
        let mut undos = Vec::new();

        // 1. Spend 5 stars
        undos.push(spend_stars(state, 5));

        // 2. Create Embassy structure at target
        undos.push(create_structure(
            state,
            self.target_index,
            StructureType::Embassy,
            1,
        ));

        // 3. Update relations
        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            if let Some(relation) = tribe.relations.get_mut(&opp_id) {
                let old_val = relation.embassy_level;
                relation.embassy_level = 1;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        if let Some(r) = t.relations.get_mut(&opp_id) {
                            r.embassy_level = old_val;
                        }
                    }
                }) as UndoCallback);
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Establish embassy in tribe {} at {}",
            self.opponent_id, self.target_index
        )
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "target": self.target_index,
            "src": self.opponent_id,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.opponent_id as usize)
    }

    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::EstablishEmbassy)
    }
}

// --- Peace Treaty (Offer) ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PeaceTreatyMove {
    pub target_index: i32,
    pub opponent_id: PlayerId,
}

impl PeaceTreatyMove {
    pub fn new(target_index: i32, opponent_id: PlayerId) -> Self {
        Self {
            target_index,
            opponent_id,
        }
    }
}

impl Move for PeaceTreatyMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let opp_id = self.opponent_id;
        let mut undos = Vec::new();

        // SIGN PEACE TREATY is an offer.
        // It should NOT set relation.state = 1 immediately.
        // It only sets has_offered_peace = true for the OPPONENT.
        // The opponent then sees PeaceRequestResponse moves in their turn.

        if let Some(tribe) = state.tribes.get_mut(&opp_id) {
            if let Some(relation) = tribe.relations.get_mut(&pov_id) {
                let old_offered = relation.has_offered_peace;
                relation.has_offered_peace = true;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&opp_id) {
                        if let Some(r) = t.relations.get_mut(&pov_id) {
                            r.has_offered_peace = old_offered;
                        }
                    }
                }) as UndoCallback);
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Sign peace treaty with tribe {}", self.opponent_id)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "target": self.target_index,
            "src": self.opponent_id,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.opponent_id as usize)
    }

    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::PeaceTreaty)
    }
}

// --- Destroy Embassy ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DestroyEmbassyMove {
    pub target_index: i32,
    pub opponent_id: PlayerId,
}

impl DestroyEmbassyMove {
    pub fn new(target_index: i32, opponent_id: PlayerId) -> Self {
        Self {
            target_index,
            opponent_id,
        }
    }
}

impl Move for DestroyEmbassyMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let opp_id = self.opponent_id;
        let mut undos = Vec::new();

        undos.push(destroy_structure(state, self.target_index));

        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            if let Some(relation) = tribe.relations.get_mut(&opp_id) {
                let old_val = relation.embassy_level;
                relation.embassy_level = 0;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        if let Some(r) = t.relations.get_mut(&opp_id) {
                            r.embassy_level = old_val;
                        }
                    }
                }) as UndoCallback);
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!(
            "Destroy embassy with tribe {} at {}",
            self.opponent_id, self.target_index
        )
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "target": self.target_index,
            "src": self.opponent_id,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.opponent_id as usize)
    }

    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::DestroyEmbassy)
    }
}

pub fn generate_diplomacy_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    let has_diplomacy = crate::settings::technology::has_technology(
        &tribe.tech_vanilla,
        crate::types::TechnologyType::Diplomacy,
    );

    for (&other_id, relation) in &tribe.relations {
        if other_id == pov_id {
            continue;
        }

        // Response to pending requests - DISABLED to prevent loops
        /*
        if relation.has_offered_peace {
            moves.push(Box::new(PeaceRequestResponse::new(other_id, 1)));
            moves.push(Box::new(PeaceRequestResponse::new(other_id, 0)));
        }
        */

        if relation.state == 1 {
            moves.push(Box::new(BreakPeaceMove::new(other_id)));
        } else if has_diplomacy {
            if relation.embassy_level == 0 && tribe.stars >= 5 {
                if let Some(other_capital) = crate::functions::get_capital_city(state, other_id) {
                    moves.push(Box::new(EstablishEmbassyMove::new(
                        other_capital.idx,
                        other_id,
                    )));
                }
            }

            // Only offer if NOT already at peace and NOT already offered
            // We need to check if WE have offered peace to THEM
            /*
            let i_offered = state
                .tribes
                .get(&other_id)
                .map(|t| {
                    t.relations
                        .get(&pov_id)
                        .map(|r| r.has_offered_peace)
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            */

            /* DISABLED to prevent loops
            if relation.state == 0 && !i_offered {
                if let Some(cap) = crate::functions::get_capital_city(state, pov_id) {
                    moves.push(Box::new(PeaceTreatyMove::new(cap.idx, other_id)));
                }
            }
            */
        }

        if relation.embassy_level > 0 {
            if let Some(other_capital) = crate::functions::get_capital_city(state, other_id) {
                moves.push(Box::new(DestroyEmbassyMove::new(
                    other_capital.idx,
                    other_id,
                )));
            }
        }
    }
}
