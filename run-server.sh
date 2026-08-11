cd polyfish-rs

# Release build: debug-profile NN search is ~100x slower and holds the game
# lock long enough to freeze the UI into queueing duplicate moves.
kill $(lsof -t -i:3000) || true && cargo run --release --bin polyfish
