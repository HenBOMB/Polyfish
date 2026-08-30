//! Ad-hoc: run EcoPlanCommit::update against a real --state dump and print
//! the credited mine-partner set. Usage: cargo run --example
//! eco_plan_commit_probe -- <state.json> <player>
use polyfish::ai::eco_plan_commit::EcoPlanCommit;
use polyfish::states::GameState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).expect("read state");
    let state: GameState = serde_json::from_str(&raw).expect("parse state");
    let player: i32 = args[2].parse().unwrap();
    let mut commit = EcoPlanCommit::default();
    commit.update(&state, player);
    for &t in &[37, 38, 39, 50] {
        println!("tile {t}: is_mine_partner = {}", commit.is_mine_partner(t));
    }
}
