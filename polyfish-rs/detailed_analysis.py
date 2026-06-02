import ast

# Load variables from output.txt
with open("output.txt", "r") as f:
    content = f.read()

# Parse the python assignments in output.txt
local_vars = {}
for line in content.split("\n"):
    if "=" in line and (line.startswith("const ") or line.startswith("const leagueItersSet = new Set")):
        # Clean JS-like syntax to python syntax
        line_clean = line.replace("const ", "")
        if "new Set" in line_clean:
            # e.g., leagueItersSet = new Set([52, 56, ...]);
            line_clean = line_clean.replace("new Set(", "").replace(");", "")
        else:
            line_clean = line_clean.rstrip(";")
        
        try:
            name, val_str = line_clean.split("=", 1)
            name = name.strip()
            val_str = val_str.strip()
            # Evaluate using ast.literal_eval for safety
            local_vars[name] = ast.literal_eval(val_str)
        except Exception as e:
            # Handle special cases or skip
            pass

losses = local_vars.get("losses", [])
avgScores = local_vars.get("avgScores", [])
maxScores = local_vars.get("maxScores", [])
p1Avgs = local_vars.get("p1Avgs", [])
p2Avgs = local_vars.get("p2Avgs", [])
avgMoves = [m for m in local_vars.get("avgMoves", []) if m is not None]
bestGames = local_vars.get("bestGames", [])
leagueItersSet = set(local_vars.get("leagueItersSet", []))

total_iters = len(losses)
print(f"Total Iterations: {total_iters}")

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

# 1. Loss Analysis
min_loss = min(losses)
min_loss_iter = losses.index(min_loss) + 1
max_loss = max(losses)
max_loss_iter = losses.index(max_loss) + 1
first_10_loss = mean(losses[:10])
last_10_loss = mean(losses[-10:])
print("\n--- LOSS METRICS ---")
print(f"Minimum Loss: {min_loss:.4f} (Iteration {min_loss_iter})")
print(f"Maximum Loss: {max_loss:.4f} (Iteration {max_loss_iter})")
print(f"First 10 Iterations Avg Loss: {first_10_loss:.4f}")
print(f"Last 10 Iterations Avg Loss: {last_10_loss:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
# Check trend
if last_10_loss > min_loss * 1.1:
    print(f"Loss has risen by {(last_10_loss - min_loss)/min_loss * 100:.1f}% from the minimum, suggesting overfitting or training instability.")
else:
    print("Loss remains stable near the minimum.")

# 2. Score Analysis
max_score = max(maxScores)
max_score_iter = maxScores.index(max_score) + 1
best_game_file = bestGames[max_score_iter - 1] if max_score_iter - 1 < len(bestGames) else None
print("\n--- SCORE METRICS ---")
print(f"Highest Score Achieved: {max_score} (Iteration {max_score_iter})")
print(f"Best Game File: {best_game_file}")
print(f"First 50 Iterations Avg Score: {mean(avgScores[:50]):.2f}")
print(f"Last 50 Iterations Avg Score: {mean(avgScores[-50:]):.2f}")
print(f"Max Avg Score: {max(avgScores):.2f} (Iteration {avgScores.index(max(avgScores)) + 1})")

# P1 vs P2 (First-mover advantage check)
p1_overall = mean(p1Avgs)
p2_overall = mean(p2Avgs)
print(f"Overall P1 Avg Score: {p1_overall:.2f}")
print(f"Overall P2 Avg Score: {p2_overall:.2f}")
print(f"Difference (P1 - P2): {p1_overall - p2_overall:.2f} ({((p1_overall - p2_overall)/p2_overall)*100:.2f}%)")

p1_wins = sum(1 for p1, p2 in zip(p1Avgs, p2Avgs) if p1 > p2)
p2_wins = sum(1 for p1, p2 in zip(p1Avgs, p2Avgs) if p2 > p1)
ties = sum(1 for p1, p2 in zip(p1Avgs, p2Avgs) if p1 == p2)
print(f"P1 scored higher in {p1_wins} iterations ({p1_wins/len(p1Avgs)*100:.1f}%)")
print(f"P2 scored higher in {p2_wins} iterations ({p2_wins/len(p2Avgs)*100:.1f}%)")

# Early vs Late P1/P2 bias
print(f"Early (First 50) P1 Avg: {mean(p1Avgs[:50]):.2f} vs P2 Avg: {mean(p2Avgs[:50]):.2f}")
print(f"Late (Last 50) P1 Avg: {mean(p1Avgs[-50:]):.2f} vs P2 Avg: {mean(p2Avgs[-50:]):.2f}")

# 3. Moves/Game Length Analysis
if avgMoves:
    print("\n--- GAME LENGTH (MOVES) METRICS ---")
    print(f"Overall Avg Moves: {mean(avgMoves):.2f}")
    print(f"First 50 Iterations Avg Moves: {mean(avgMoves[:50]):.2f}")
    print(f"Last 50 Iterations Avg Moves: {mean(avgMoves[-50:]):.2f}")
    print(f"Min Avg Moves: {min(avgMoves):.2f} (Iteration {avgMoves.index(min(avgMoves)) + 1})")
    print(f"Max Avg Moves: {max(avgMoves):.2f} (Iteration {avgMoves.index(max(avgMoves)) + 1})")

# 4. League Matches Analysis
league_losses = []
league_scores = []
league_moves = []
self_play_losses = []
self_play_scores = []
self_play_moves = []

for i in range(total_iters):
    iter_num = i + 1
    if iter_num in leagueItersSet:
        league_losses.append(losses[i])
        league_scores.append(avgScores[i])
        if i < len(avgMoves):
            league_moves.append(avgMoves[i])
    else:
        self_play_losses.append(losses[i])
        self_play_scores.append(avgScores[i])
        if i < len(avgMoves):
            self_play_moves.append(avgMoves[i])

print("\n--- LEAGUE VS SELF-PLAY ANALYSIS ---")
print(f"Total League Iterations: {len(league_losses)} ({len(league_losses)/total_iters*100:.1f}%)")
print(f"Total Self-Play Iterations: {len(self_play_losses)} ({len(self_play_losses)/total_iters*100:.1f}%)")
print(f"League Avg Loss: {mean(league_losses):.4f} vs Self-Play Avg Loss: {mean(self_play_losses):.4f}")
print(f"League Avg Score: {mean(league_scores):.2f} vs Self-Play Avg Score: {mean(self_play_scores):.2f}")
if league_moves and self_play_moves:
    print(f"League Avg Moves: {mean(league_moves):.2f} vs Self-Play Avg Moves: {mean(self_play_moves):.2f}")
