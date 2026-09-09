# PolyfishAI Mod

This is a mod for **The Battle of Polytopia** that helps capture replay data for the PolyAI project. 

Currently it basically sits inside the game and "scrapes" what happens in replays (like what an explorer finds or what rewards come from ruins) so we can use that data to train the AI.

## 🛠 What it does
- **Auto-Replay**: Opens and runs through replays on its own.
- **Fast Forward**: Logic speedup (up to 40x; experimental and currently commented out in `src/PolyfishAI.cs:42` to maintain animation timing stability during capture).
- **Data Capture**: Sends game states and moves to a local server.

## 📥 How to use it
1. Make sure you have **PolyMod** or **BepInEx** installed.
2. Put the `.dll` (compiled code) in your game's `plugins` folder.
3. Keep the **PolyAI/polyfish-rs** server running in the background while the game is open.
