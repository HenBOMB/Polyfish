@echo off
cd /d "%~dp0polyfish-rs"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /f /pid %%a 2>nul
)
cargo run --bin polyfish
