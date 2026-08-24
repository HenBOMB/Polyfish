#!/usr/bin/env python3
"""Emit a standalone Rust probe that counts legal moves per turn.

Scaffolding for the branching-factor analysis written up in notes.md — it
writes /tmp/test_branching.rs for you to compile against the crate; it does not
run anything itself. Lived in tests/ under a test_* name, where `unittest
discover` picked it up and executed it as a test case.
"""

import subprocess
import json

# We'll create a simple Rust program that plays a game and logs move counts
rust_test = """
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType};
use polyfish::TribeType;

fn main() {
    let gen_settings = MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 12345,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.state.settings.mode = ModeType::Perfection;
    game.state.settings.max_turns = 10;
    game.post_load();

    println!("Starting 10-turn Perfection game...");
    println!("Turn,Player,LegalMoves");

    let mut turn_count = 0;
    while !polyfish::functions::is_game_over(&game.state) && turn_count < 100 {
        let legal_moves = game.legal_moves();
        let current_player = game.state.settings.current_player_turn_id;
        let current_turn = game.state.settings.turn;
        
        println!("{},{},{}", current_turn, current_player, legal_moves.len());
        
        // Play first legal move (simple strategy)
        if let Some(m) = legal_moves.first() {
            let _ = game.play_move(m.as_ref());
            turn_count += 1;
        } else {
            break;
        }
    }
    
    println!("Game ended after {} moves", turn_count);
}
"""

with open("/tmp/test_branching.rs", "w") as f:
    f.write(rust_test)

print("Created test script. Compile and run...")
