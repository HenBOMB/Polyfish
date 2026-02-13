use crate::game::Game;
use crate::moves::Move;
use crate::types::{MoveType, TribeType};

pub struct Book;

impl Book {
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

    // TODO most tribes dont use these openings
    fn get_book_moves(tribe: TribeType, turn: i32) -> &'static [MoveType] {
        match tribe {
            TribeType::Imperius => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Summon, MoveType::Step],
                _ => &[],
            },
            TribeType::Bardur | TribeType::Kickoo | TribeType::Zebasi | TribeType::Yadakk => {
                match turn {
                    0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                    1 => &[MoveType::Summon, MoveType::Step],
                    _ => &[],
                }
            }
            TribeType::XinXi | TribeType::Oumaji => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Luxidoor => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Vengir => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::AiMo => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Quetzali => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Hoodrick => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Elyrion => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward], // Enchanct? Harvest logic might cover EnchantAnimal
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Polaris => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward], // Mooni move?
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Cymanti => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward], // Fungi?
                1 => &[MoveType::Step],
                _ => &[],
            },
            TribeType::Aquarion => match turn {
                0 => &[MoveType::Harvest, MoveType::Step, MoveType::Reward],
                1 => &[MoveType::Summon, MoveType::Step],
                _ => &[],
            },
            _ => &[],
        }
    }
}
