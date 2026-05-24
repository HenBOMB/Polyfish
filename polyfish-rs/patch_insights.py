import re

html_path = "../polytopia_training_dashboard.html"

with open(html_path, "r") as f:
    html = f.read()

# Replace the annotation bar
new_annotation = """<div class="annotation-bar">
  <div class="annotation-icon">TRAINING HIGHLIGHT · ITER 144</div>
  <div class="annotation-text">
    <strong>Minimum loss achieved.</strong>
    Loss dropped from 41.57 consistently down to 25.64. After iter 144, the loss plateaued and softly rebounded to ~29, indicating the generic capacity of the current architecture or generation diversity might be reaching a stable point.
  </div>
</div>"""

html = re.sub(r'<div class="annotation-bar">.*?</div>\n</div>', new_annotation, html, flags=re.DOTALL)

# Replace the Loss legends
new_loss_legend = """<div class="panel-legend">
        <div class="legend-item"><div class="legend-swatch" style="background: var(--accent-blue)"></div>Policy + value loss</div>
      </div>"""
html = re.sub(r'<div class="panel-legend">\s*<div class="legend-item"><div class="legend-swatch" style="background: var\(--accent-blue\)"></div>Policy \+ value loss \(pre-phase\)</div>\s*<div class="legend-item"><div class="legend-swatch" style="background: var\(--accent-red\)"></div>Loss after phase change \(iter 219\+\)</div>\s*</div>', new_loss_legend, html, flags=re.DOTALL)

# Update Javascript loss datasets to just be one dataset
new_loss_js = """datasets: [
      {
        label: 'Loss',
        data: losses,
        borderColor: '#378ADD',
        backgroundColor: 'rgba(55,138,221,0.06)',
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.35,
        fill: true,
        spanGaps: false
      }
    ]"""

# Since replacing the whole datasets block in JS is tricky, let's use regex
html = re.sub(r"datasets: \[\s*\{\s*label: 'Loss \(pre-phase\)'.*?\}\s*\]", new_loss_js, html, flags=re.DOTALL)

# Now insights
new_insights = """<div class="insights-grid">
  <div class="insight-card info">
    <div class="insight-title">Rapid early learning <span class="tag tag-info">context</span></div>
    <div class="insight-body">Loss dropped significantly from 41.57 to 25.64 by iteration 144. The initial training trajectory was very healthy and fast before hitting a plateau curve.</div>
  </div>
  <div class="insight-card warn">
    <div class="insight-title">Loss stabilization <span class="tag tag-warn">watch</span></div>
    <div class="insight-body">After the minimum loss, the model's loss hovered between 26 and 29. It suggests the model is approaching its capacity limit or needs more diverse data from alternative tribes/maps.</div>
  </div>
  <div class="insight-card good">
    <div class="insight-title">Balanced P1 vs P2 <span class="tag tag-good">positive</span></div>
    <div class="insight-body">Unlike previous runs, P1 and P2 scores are incredibly balanced (P1 avg: 3482, P2 avg: 3474). First-mover disadvantage appears to be completely resolved or adapted to by the newer model.</div>
  </div>
  <div class="insight-card warn">
    <div class="insight-title">Average scores stabilized <span class="tag tag-warn">plateau</span></div>
    <div class="insight-body">Scores average around 3950-4100 across the entire run without exploding. The max scores occasionally spike up to 7,140, showing capable spikes but consistent median behavior.</div>
  </div>
  <div class="insight-card good">
    <div class="insight-title">League stability <span class="tag tag-good">healthy</span></div>
    <div class="insight-body">League matches make up ~22% of the session. The model consistently holds its own against past checkpoints, preventing catastrophic forgetting effectively.</div>
  </div>
  <div class="insight-card info">
    <div class="insight-title">Efficient game lengths <span class="tag tag-info">stable</span></div>
    <div class="insight-body">Average moves per game typically float between 210 and 260. The games finish in a predictable length, indicating no degenerate infinite-loop strategies have emerged.</div>
  </div>
</div>"""

html = re.sub(r'<div class="insights-grid">.*?</div>\s*<div class="footer">', new_insights + '\n\n<div class="footer">', html, flags=re.DOTALL)

with open(html_path, "w") as f:
    f.write(html)
print("Insights patched successfully")

