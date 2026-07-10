cd polyfish-rs

pids=$(lsof -t -i:3000)
if [ -n "$pids" ]; then
    kill $pids
fi
cargo run --bin polyfish
