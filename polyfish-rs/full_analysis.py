import csv
import json
import re
from datetime import datetime

def mean(lst):
    return sum(lst) / len(lst) if lst else 0

def median(lst):
    s = sorted(lst)
    n = len(s)
    if n == 0: return 0
    if n % 2 == 0:
        return (s[n//2-1] + s[n//2]) / 2
    return s[n//2]

# =============================================
# 1. Parse training_log.csv (canonical data)
# =============================================
rows = []
with open("training_log.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append({
            "iter": int(row[0]),
            "timestamp": int(row[1]),
            "avg_score": float(row[2]),
            "max_score": int(row[3]),
            "p1_avg": float(row[4]),
            "p2_avg": float(row[5]),
            "loss": float(row[6]),
            "avg_captures": float(row[7]),
            "avg_harvests": float(row[8]),
            "avg_builds": float(row[9]),
            "avg_research": float(row[10]),
            "avg_attacks": float(row[11]),
        })

n = len(rows)

# =============================================
# 2. Parse session.log for loss + league data
# =============================================
session_losses = []
session_scores = []
league_iters = []

try:
    with open("session.log", "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_iter = 0
    for line in lines:
        if line.startswith("Starting Iteration"):
            m = re.search(r"Starting Iteration (\d+)", line)
            if m:
                current_iter = int(m.group(1))
        elif "[League Match" in line:
            league_iters.append(current_iter)
        elif line.startswith("METRICS: "):
            try:
                data = json.loads(line.replace("METRICS: ", ""))
                if "loss" in data and "avg_score" not in data:
                    session_losses.append(data["loss"])
            except:
                pass
except Exception as e:
    print(f"Warning: Could not parse session.log: {e}")

# Deduplicate league iters
league_set = set(league_iters)

# =============================================
# Extract all metric arrays
# =============================================
scores = [r["avg_score"] for r in rows]
max_scores = [r["max_score"] for r in rows]
p1 = [r["p1_avg"] for r in rows]
p2 = [r["p2_avg"] for r in rows]
caps = [r["avg_captures"] for r in rows]
harv = [r["avg_harvests"] for r in rows]
blds = [r["avg_builds"] for r in rows]
rsch = [r["avg_research"] for r in rows]
atks = [r["avg_attacks"] for r in rows]

# =============================================
# PRINT ANALYSIS
# =============================================
print("=" * 60)
print("   POLYFISH TRAINING DATA ANALYSIS")
print("=" * 60)

print(f"\nTotal iterations: {n}")
print(f"Iteration range: {rows[0]['iter']} -> {rows[-1]['iter']}")

t0 = datetime.fromtimestamp(rows[0]["timestamp"])
t1 = datetime.fromtimestamp(rows[-1]["timestamp"])
duration = t1 - t0
print(f"Training start: {t0.strftime('%Y-%m-%d %H:%M')}")
print(f"Training end:   {t1.strftime('%Y-%m-%d %H:%M')}")
print(f"Total duration: {duration}")
hours = duration.total_seconds() / 3600
print(f"Avg time/iter:  {hours/n*60:.1f} minutes")

# =============================================
# SCORE ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("   SCORE ANALYSIS")
print("=" * 60)

print(f"\nOverall avg score:  {mean(scores):.1f}")
print(f"Overall median:     {median(scores):.1f}")
print(f"Overall std dev:    {(sum((s-mean(scores))**2 for s in scores)/n)**0.5:.1f}")
print(f"\nFirst 50 iters avg: {mean(scores[:50]):.1f}")
print(f"Mid 50 iters avg:   {mean(scores[50:100]):.1f} (iters 51-100)")
print(f"Last 50 iters avg:  {mean(scores[-50:]):.1f}")
score_improvement = ((mean(scores[-50:]) - mean(scores[:50])) / mean(scores[:50])) * 100
print(f"Improvement (first50 -> last50): +{score_improvement:.1f}%")

print(f"\nMax single game score: {max(max_scores)}")
best_max_idx = max_scores.index(max(max_scores))
print(f"  -> At iteration {rows[best_max_idx]['iter']} (avg that iter: {rows[best_max_idx]['avg_score']:.0f})")
print(f"Max avg score in any iter: {max(scores):.0f} at iter {rows[scores.index(max(scores))]['iter']}")

# Score trajectory
print("\n--- Score Progression (25-iter windows) ---")
for i in range(0, n, 25):
    chunk = scores[i:i+25]
    mx = max_scores[i:i+25]
    if chunk:
        print(f"  Iters {rows[i]['iter']:3d}-{rows[min(i+24, n-1)]['iter']:3d}: avg={mean(chunk):7.1f}  max={max(mx):5d}  min_avg={min(chunk):7.1f}")

# =============================================
# P1 vs P2 BALANCE
# =============================================
print("\n" + "=" * 60)
print("   P1 vs P2 BALANCE (First-Mover Analysis)")
print("=" * 60)

p1_mean = mean(p1)
p2_mean = mean(p2)
print(f"\nP1 overall avg: {p1_mean:.1f}")
print(f"P2 overall avg: {p2_mean:.1f}")
diff_pct = ((p1_mean - p2_mean) / p2_mean) * 100
print(f"Difference: {p1_mean - p2_mean:+.1f} ({diff_pct:+.2f}%)")

p1_wins = sum(1 for a, b in zip(p1, p2) if a > b)
p2_wins = sum(1 for a, b in zip(p1, p2) if b > a)
ties = sum(1 for a, b in zip(p1, p2) if a == b)
print(f"\nP1 scored higher: {p1_wins} iters ({p1_wins/n*100:.1f}%)")
print(f"P2 scored higher: {p2_wins} iters ({p2_wins/n*100:.1f}%)")
print(f"Equal: {ties} iters")

print(f"\n--- P1/P2 by Phase ---")
print(f"  First 50:  P1={mean(p1[:50]):.0f} vs P2={mean(p2[:50]):.0f}  (diff: {mean(p1[:50])-mean(p2[:50]):+.0f})")
print(f"  Iters 51-100: P1={mean(p1[50:100]):.0f} vs P2={mean(p2[50:100]):.0f}  (diff: {mean(p1[50:100])-mean(p2[50:100]):+.0f})")
print(f"  Iters 101-150: P1={mean(p1[100:150]):.0f} vs P2={mean(p2[100:150]):.0f}  (diff: {mean(p1[100:150])-mean(p2[100:150]):+.0f})")
print(f"  Last 50:   P1={mean(p1[-50:]):.0f} vs P2={mean(p2[-50:]):.0f}  (diff: {mean(p1[-50:])-mean(p2[-50:]):+.0f})")

# =============================================
# LOSS ANALYSIS (from session.log)
# =============================================
print("\n" + "=" * 60)
print("   LOSS ANALYSIS")
print("=" * 60)

if session_losses:
    print(f"\nTotal loss entries from session.log: {len(session_losses)}")
    print(f"Starting loss: {session_losses[0]:.4f}")
    print(f"Minimum loss:  {min(session_losses):.4f} (at training step {session_losses.index(min(session_losses))+1})")
    print(f"Final loss:    {session_losses[-1]:.4f}")
    print(f"Loss reduction: {((session_losses[0] - min(session_losses))/session_losses[0]*100):.1f}%")
    rebound = ((session_losses[-1] - min(session_losses)) / min(session_losses)) * 100
    print(f"Loss rebound from min: +{rebound:.1f}%")

    print(f"\nFirst 10 losses avg: {mean(session_losses[:10]):.4f}")
    print(f"Last 10 losses avg:  {mean(session_losses[-10:]):.4f}")

    # Loss trend
    print("\n--- Loss Progression ---")
    chunk_size = max(1, len(session_losses) // 10)
    for i in range(0, len(session_losses), chunk_size):
        chunk = session_losses[i:i+chunk_size]
        if chunk:
            print(f"  Steps {i+1:3d}-{min(i+chunk_size, len(session_losses)):3d}: avg={mean(chunk):.4f} min={min(chunk):.4f} max={max(chunk):.4f}")
else:
    print("\nNote: All losses in training_log.csv are 0.0")
    print("Loss data is only available in session.log METRICS lines")

# =============================================
# GAMEPLAY EVOLUTION
# =============================================
print("\n" + "=" * 60)
print("   GAMEPLAY BEHAVIOR EVOLUTION")
print("=" * 60)

metrics = {
    "Captures": caps,
    "Harvests": harv,
    "Builds": blds,
    "Research": rsch,
    "Attacks": atks,
}

for name, vals in metrics.items():
    early = mean(vals[:50])
    late = mean(vals[-50:])
    change = ((late - early) / early * 100) if early > 0 else float("inf")
    print(f"\n--- {name} ---")
    print(f"  Early (1-50):  {early:.2f}")
    print(f"  Late (last 50): {late:.2f}")
    if abs(change) != float("inf"):
        print(f"  Change: {change:+.1f}%")
    else:
        print(f"  Change: N/A (early was 0)")
    print(f"  Overall avg: {mean(vals):.2f}")
    print(f"  Max: {max(vals):.2f} at iter {rows[vals.index(max(vals))]['iter']}")

# =============================================
# LEAGUE ANALYSIS
# =============================================
print("\n" + "=" * 60)
print("   LEAGUE vs SELF-PLAY ANALYSIS")
print("=" * 60)

league_rows = [r for r in rows if r["iter"] in league_set]
selfplay_rows = [r for r in rows if r["iter"] not in league_set]

print(f"\nLeague iterations: {len(league_rows)} ({len(league_rows)/n*100:.1f}%)")
print(f"Self-play iterations: {len(selfplay_rows)} ({len(selfplay_rows)/n*100:.1f}%)")

if league_rows and selfplay_rows:
    l_scores = [r["avg_score"] for r in league_rows]
    s_scores = [r["avg_score"] for r in selfplay_rows]
    print(f"\nLeague avg score:    {mean(l_scores):.1f}")
    print(f"Self-play avg score: {mean(s_scores):.1f}")
    print(f"Difference: {mean(l_scores) - mean(s_scores):+.1f}")

    l_caps = [r["avg_captures"] for r in league_rows]
    s_caps = [r["avg_captures"] for r in selfplay_rows]
    print(f"\nLeague avg captures:    {mean(l_caps):.2f}")
    print(f"Self-play avg captures: {mean(s_caps):.2f}")

    l_atks = [r["avg_attacks"] for r in league_rows]
    s_atks = [r["avg_attacks"] for r in selfplay_rows]
    print(f"League avg attacks:     {mean(l_atks):.2f}")
    print(f"Self-play avg attacks:  {mean(s_atks):.2f}")

# =============================================
# CURRICULUM / PHASE DETECTION
# =============================================
print("\n" + "=" * 60)
print("   PHASE / CURRICULUM DETECTION")
print("=" * 60)

# Detect score jumps using moving average
window = 10
jumps = []
for i in range(window, n):
    prev_avg = mean(scores[i-window:i])
    if scores[i] > prev_avg * 1.4 and prev_avg > 0:
        jumps.append((rows[i]["iter"], prev_avg, scores[i]))

if jumps:
    print("\nSignificant score jumps detected:")
    for it, prev, curr in jumps:
        print(f"  Iter {it}: {prev:.0f} -> {curr:.0f} ({(curr/prev-1)*100:+.0f}%)")
else:
    print("\nNo dramatic score jumps detected - smooth progression")

# Phase boundaries by score thresholds
print("\n--- Score Phase Summary ---")
phases = [(0, 50), (50, 100), (100, 150), (150, 200), (200, n)]
for s, e in phases:
    chunk = rows[s:e]
    sc = [r["avg_score"] for r in chunk]
    mx = [r["max_score"] for r in chunk]
    cp = [r["avg_captures"] for r in chunk]
    at = [r["avg_attacks"] for r in chunk]
    hr = [r["avg_harvests"] for r in chunk]
    bl = [r["avg_builds"] for r in chunk]
    rs = [r["avg_research"] for r in chunk]
    print(f"  Iters {chunk[0]['iter']:3d}-{chunk[-1]['iter']:3d}: Score={mean(sc):7.0f} Max={max(mx):5d} Caps={mean(cp):.1f} Atk={mean(at):.1f} Harv={mean(hr):.1f} Build={mean(bl):.1f} Tech={mean(rs):.1f}")

# =============================================
# TOP/BOTTOM ITERATIONS
# =============================================
print("\n" + "=" * 60)
print("   TOP 10 HIGHEST SCORING ITERATIONS")
print("=" * 60)

sorted_top = sorted(rows, key=lambda r: r["max_score"], reverse=True)[:10]
for r in sorted_top:
    league_tag = " [LEAGUE]" if r["iter"] in league_set else ""
    print(f"  Iter {r['iter']:3d}: max={r['max_score']:5d} avg={r['avg_score']:7.1f} p1={r['p1_avg']:7.1f} p2={r['p2_avg']:7.1f}{league_tag}")

print("\n" + "=" * 60)
print("   BOTTOM 10 LOWEST SCORING ITERATIONS")
print("=" * 60)

sorted_bot = sorted(rows, key=lambda r: r["avg_score"])[:10]
for r in sorted_bot:
    league_tag = " [LEAGUE]" if r["iter"] in league_set else ""
    print(f"  Iter {r['iter']:3d}: avg={r['avg_score']:7.1f} max={r['max_score']:5d} caps={r['avg_captures']:.1f} atk={r['avg_attacks']:.1f}{league_tag}")

# =============================================
# TRAINING HEALTH INDICATORS
# =============================================
print("\n" + "=" * 60)
print("   TRAINING HEALTH INDICATORS")
print("=" * 60)

# Variance analysis
score_var_early = sum((s - mean(scores[:50]))**2 for s in scores[:50]) / 50
score_var_late = sum((s - mean(scores[-50:]))**2 for s in scores[-50:]) / 50
print(f"\nScore variance (first 50):  {score_var_early:.0f}")
print(f"Score variance (last 50):   {score_var_late:.0f}")
print(f"Variance ratio: {score_var_late/score_var_early:.2f}x")

# P1/P2 gap trend
p1p2_gaps = [abs(a-b) for a,b in zip(p1, p2)]
print(f"\nP1-P2 gap (first 50):  {mean(p1p2_gaps[:50]):.1f}")
print(f"P1-P2 gap (last 50):   {mean(p1p2_gaps[-50:]):.1f}")

# Monotonicity check (is score consistently rising?)
rising = 0
falling = 0
window = 20
for i in range(window, n):
    prev = mean(scores[i-window:i])
    curr = scores[i]
    if curr > prev:
        rising += 1
    else:
        falling += 1
print(f"\nScore above rolling avg: {rising} ({rising/(rising+falling)*100:.0f}%)")
print(f"Score below rolling avg: {falling} ({falling/(rising+falling)*100:.0f}%)")

# Plateau detection
last_100_trend = (mean(scores[-25:]) - mean(scores[-100:-75]))
print(f"\nLast 100 iters trend (last25 - first25 of last100): {last_100_trend:+.0f}")
if abs(last_100_trend) < mean(scores[-100:]) * 0.05:
    print("  -> PLATEAU detected: scores have stabilized")
elif last_100_trend > 0:
    print("  -> RISING: model is still improving")
else:
    print("  -> DECLINING: possible regression or overfitting")

# Archive data size
print(f"\n--- Data Volume ---")
total_archive_gb = sum([
    233127672, 335908232, 305975112, 259471872, 286426952,
    281998072, 300171752, 207776152, 379967952, 310556712,
    192351432, 273445752, 276041992, 296277392, 228927872,
    287801432, 280470872, 271765832, 283143472, 285357912,
    291924872, 311931192, 295284712, 267718752, 345529592,
    278790952, 275354752, 293146632, 283677992, 218542912
]) / (1024**3)
print(f"Active replay buffer: 30 files ({total_archive_gb:.2f} GB)")
print(f"Model checkpoints: 4 (iters 50, 100, 150, 200)")

print("\n" + "=" * 60)
print("   ANALYSIS COMPLETE")
print("=" * 60)
