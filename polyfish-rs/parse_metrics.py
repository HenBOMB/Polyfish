import json
import re

losses = []
avgScores = []
maxScores = []
p1Avgs = []
p2Avgs = []
avgMoves = []
bestGames = []

leagueItersSet = []

# One (iteration, moves_by_turn dict) pair per self-play METRICS line, used to
# sample the "move mix by turn" training-progress chart.
movesByTurnEntries = []

with open("session.log", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

iter_count = 0
current_iter_metrics = {}

for i, line in enumerate(lines):
    if line.startswith("Starting Iteration"):
        iter_count = int(re.search(r"Starting Iteration (\d+)", line).group(1))
    elif "[League Match" in line:
        leagueItersSet.append(iter_count)
    elif line.startswith("METRICS: "):
        try:
            data = json.loads(line.replace("METRICS: ", ""))
            if "loss" in data:
                losses.append(data["loss"])
            if "avg_score" in data:
                avgScores.append(data["avg_score"])
                maxScores.append(data["max_score"])
                avgMoves.append(data.get("avg_moves", "null"))
                p1Avgs.append(data["p1_avg"])
                p2Avgs.append(data["p2_avg"])
            if "moves_by_turn" in data and data["moves_by_turn"]:
                movesByTurnEntries.append((iter_count, data["moves_by_turn"]))
        except Exception:
            pass
    elif "🏆 Highest score game" in line:
        match = re.search(r"saved to (.*)", line)
        if match:
            # Replaces full path with relative path for dashboard
            path = match.group(1).replace("/home/henry/Desktop/Coding/PolyAI/polyfish-rs/", "")
            bestGames.append(path)
    # Ensure bestGames stays in sync with iterations if some iterations don't save a game (unlikely with my change but safe)
    # Actually, self_play always prints it now.

print("const losses = " + str(losses) + ";")
print("const avgScores = " + str(avgScores) + ";")
print("const maxScores = " + str(maxScores) + ";")
print("const p1Avgs = " + str(p1Avgs) + ";")
print("const p2Avgs = " + str(p2Avgs) + ";")

# formatting avgMoves to include null properly
avgMovesStr = "[" + ",".join(str(m) if m != "null" else "null" for m in avgMoves) + "]"
print("const avgMoves = " + str(avgMovesStr) + ";")

# Ensure bestGames matches count if needed (pad with None if missing)
while len(bestGames) < len(losses):
    bestGames.append(None)

bestGamesStr = "[" + ",".join(f'"{g}"' if g else "null" for g in bestGames) + "]"
print("const bestGames = " + bestGamesStr + ";")

print("const leagueItersSet = new Set(" + str(leagueItersSet) + ");")
print("const totalIters = " + str(iter_count) + ";")

# Sample ~10 iterations evenly across the full training run (first through
# most recent) so the "move mix by turn" chart shows the progress journey
# rather than one noisy snapshot.
NUM_SAMPLES = 10
n = len(movesByTurnEntries)
if n == 0:
    sampled = []
elif n <= NUM_SAMPLES:
    sampled = movesByTurnEntries
else:
    idxs = sorted({round(i * (n - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)})
    sampled = [movesByTurnEntries[i] for i in idxs]

moveTurnSamples = [{"iter": it, "data": mbt} for it, mbt in sampled]
print("const moveTurnSamples = " + json.dumps(moveTurnSamples) + ";")

