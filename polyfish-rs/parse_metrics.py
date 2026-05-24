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

with open("session.log", "r") as f:
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

