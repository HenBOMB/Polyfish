import json

with open("output.txt") as f:
    lines = f.readlines()

losses = eval(lines[0].split("=")[1].strip()[:-1])
avgScores = eval(lines[1].split("=")[1].strip()[:-1])
maxScores = eval(lines[2].split("=")[1].strip()[:-1])
p1Avgs = eval(lines[3].split("=")[1].strip()[:-1])
p2Avgs = eval(lines[4].split("=")[1].strip()[:-1])

print(f"Min loss: {min(losses):.4f} at iter {losses.index(min(losses)) + 1}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Max score: {max(maxScores)} at iter {maxScores.index(max(maxScores)) + 1}")
print(f"Avg P1 score: {sum(p1Avgs)/len(p1Avgs):.2f}")
print(f"Avg P2 score: {sum(p2Avgs)/len(p2Avgs):.2f}")

# Look at loss shape
min_loss_idx = losses.index(min(losses))
print(f"Loss climbs from {losses[min_loss_idx]:.4f} at iter {min_loss_idx + 1} to {losses[-1]:.4f} at iter {len(losses)}")

wins_p1 = sum(1 for p1, p2 in zip(p1Avgs, p2Avgs) if p1 > p2)
wins_p2 = sum(1 for p1, p2 in zip(p1Avgs, p2Avgs) if p2 > p1)
print(f"P1 higher avg in {wins_p1} iters, P2 higher in {wins_p2} iters")

print(f"Avg Score shape: first 50 iters: {sum(avgScores[:50])/50:.1f}, last 50 iters: {sum(avgScores[-50:])/50:.1f}")
print(f"Avg Score max plateau: {max(avgScores)}")

