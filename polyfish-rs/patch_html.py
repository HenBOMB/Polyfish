import re
import os

# 1. Read metrics from output.txt
metrics_file = "output.txt"
if not os.path.exists(metrics_file):
    print(f"Error: {metrics_file} not found")
    exit(1)

with open(metrics_file) as f:
    output_lines = [line.strip() for line in f.readlines()]

# Expected order from parse_metrics.py:
# 0: losses, 1: avgScores, 2: maxScores, 3: p1Avgs, 4: p2Avgs, 5: avgMoves, 6: bestGames,
# 7: leagueItersSet, 8: totalIters, 9: moveTurnSamples
new_data = {
    "losses": output_lines[0],
    "avgScores": output_lines[1],
    "maxScores": output_lines[2],
    "p1Avgs": output_lines[3],
    "p2Avgs": output_lines[4],
    "avgMoves": output_lines[5],
    "bestGames": output_lines[6],
    "leagueItersSet": output_lines[7],
    "moveTurnSamples": output_lines[9] if len(output_lines) > 9 else "const moveTurnSamples = [];"
}

total_match = re.search(r"totalIters = (\d+)", output_lines[8])
total_iters = total_match.group(1) if total_match else "232"

# 2. Read HTML
html_path = "../polytopia_training_dashboard.html"
with open(html_path, "r") as f:
    html = f.read()

# 3. Replace data arrays
for var_name, replacement in new_data.items():
    pattern = rf"const {var_name} = \[.*?\];"
    if var_name == "leagueItersSet":
        pattern = r"const leagueItersSet = new Set\(\[.*?\]\);"
    
    html = re.sub(pattern, replacement, html, flags=re.DOTALL)

# 4. Updates for Iteration count
html = re.sub(r"const iterations = Array\.from\(\{length: \d+\}, \(_, i\) => \d+ \+ i\);",
              f"const iterations = Array.from({{length: {total_iters}}}, (_, i) => 1 + i);", html)

# Header
html = re.sub(r"ITER 1 → \d+ &nbsp;·&nbsp; \d+ ITERATIONS", f"ITER 1 → {total_iters} &nbsp;·&nbsp; {total_iters} ITERATIONS", html)

# Metric card for total iters
html = re.sub(r'<div class="metric-value">\d+</div>', f'<div class="metric-value">{total_iters}</div>', html, count=1)

# Write back
with open(html_path, "w") as f:
    f.write(html)

print(f"Successfully patched dashboard for {total_iters} iterations.")

