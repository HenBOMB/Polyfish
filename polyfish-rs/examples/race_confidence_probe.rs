//! Ad-hoc: check village_race_confidence against a real --state dump.
//! Usage: cargo run --example race_confidence_probe -- <state.json> <player> <village_tile>
use polyfish::ai::movement::village_race_confidence;
use polyfish::states::GameState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read state");
    let state: GameState = serde_json::from_str(&raw).expect("parse state");
    let player: i32 = args[2].parse().unwrap();
    let village: i32 = args[3].parse().unwrap();
    let conf = village_race_confidence(&state, player, village);
    println!("village_race_confidence(player={player}, village={village}) = {conf:.3}");
}
