use crate::game::Game;
use crate::moves::Move;
use crate::types::{MoveType, TribeType};

pub struct Opening;

impl Opening {
    /// Returns a list of recommend moves from the opening book.
    /// Returns empty vector if no book moves are found.
    pub fn recommend(game: &Game) -> Vec<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let tribe = match game.state.tribes.get(&pov) {
            Some(t) => t.tribe_type,
            None => return Vec::new(),
        };
        let turn = game.state.settings.turn;

        // Get preferred move types for this tribe and turn
        let preferred_types = Self::get_book_moves(tribe, turn);

        if preferred_types.is_empty() {
            return Vec::new();
        }

        let legal_moves = game.legal_moves();
        if legal_moves.len() < 2 {
            // If only 1 move (e.g. EndTurn or forced), no need for book
            return Vec::new();
        }

        // Filter legal moves that match preferred types
        // We preserve order of preferred_types (priority) if we wanted,
        // but the TS implementation just collected all matches.
        let mut recommended = Vec::new();

        for m in legal_moves {
            if preferred_types.contains(&m.move_type()) {
                recommended.push(m);
            }
        }

        recommended
    }

    fn get_book_moves(tribe: TribeType, turn: i32) -> &'static [MoveType] {
        match tribe {
            TribeType::Imperius => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Summon, MoveType::Step], // In Rust game turns are 1-based, TS might have been 0-based.
                // TS: 0 -> Harvest, 1 -> Summon.
                // In Polyfish Rust: Turn starts at 1.
                // So TS Turn 0 is Rust Turn 1.
                _ => &[],
            },
            TribeType::Bardur | TribeType::Kickoo | TribeType::Zebasi | TribeType::Yadakk => {
                match turn {
                    1 => &[MoveType::Harvest, MoveType::Step],
                    2 => &[MoveType::Summon, MoveType::Step],
                    _ => &[],
                }
            }
            TribeType::XinXi | TribeType::Oumaji => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            // Fallback for others to generic "Safe" opening if desired,
            // or just copy strictly from TS.
            // TS had explicit entries for all.
            TribeType::Luxidoor => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Vengir => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::AiMo => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Quetzali => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Hoodrick => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Elyrion => match turn {
                1 => &[MoveType::Harvest, MoveType::Step], // Enchanct? Harvest logic might cover EnchantAnimal
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Polaris => match turn {
                1 => &[MoveType::Harvest, MoveType::Step], // Mooni move?
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Cymanti => match turn {
                1 => &[MoveType::Harvest, MoveType::Step], // Fungi?
                2 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Aquarion => match turn {
                1 => &[MoveType::Harvest, MoveType::Step],
                2 => &[MoveType::Summon, MoveType::Step],
                _ => &[],
            },
            _ => &[],
        }
    }

    /// Returns a list of forbidden move types for the current turn.
    /// These moves should be pruned from MCTS expansion.
    pub fn prohibited(game: &Game) -> Vec<MoveType> {
        let turn = game.state.settings.turn;
        let mut forbidden = Vec::new();

        // User Rule: On turn 2, NEVER research tech containing
        if turn == 2 {
            forbidden.push(MoveType::Research);
        }

        forbidden
    }
}
