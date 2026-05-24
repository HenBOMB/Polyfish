import re

with open("output.txt") as f:
    output_lines = f.readlines()

new_losses = output_lines[0].strip()
new_avgScores = output_lines[1].strip()
new_maxScores = output_lines[2].strip()
new_p1Avgs = output_lines[3].strip()
new_p2Avgs = output_lines[4].strip()
new_avgMoves = output_lines[5].strip()
new_leagueItersSet = output_lines[6].strip()

with open("polytopia_training_dashboard.html") as f:
    html = f.read()

# I will find the exact blocks in html to replace.
# Data arrays start with const losses = ...
