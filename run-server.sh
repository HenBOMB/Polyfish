cd polyfish-rs

kill $(lsof -t -i:3000) || true && cargo run --bin polyfish --release
