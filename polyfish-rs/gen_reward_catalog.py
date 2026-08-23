#!/usr/bin/env python3
"""Regenerates the reward catalog artifact from source, not from memory.

Parses every `pub const` (with its preceding `///` doc comment) out of the
files that actually define reward-shaping magnitudes -- goal_shape_consts.rs
(the dominant, always-live T3 goal-priced Phi), dev_potential.rs (T1/T2
label-side Phi, CLI-gated, off by default), and economy_completion.rs (a
shared helper constant goal_potential.rs imports). Everything else on the
page (scoring.rs's inline heuristic weights, the self_play CLI label/tree
knobs, the evaluator/* leaf heuristic) is hand-curated and clearly marked as
such -- see CURATED_* below -- because it either isn't a `pub const` at a
stable location (scoring.rs) or already has a canonical source of truth that
would drift if duplicated here (`self_play --help`).

Usage: python3 gen_reward_catalog.py > /path/to/output.html
"""
import re
import subprocess
import sys
import os
import html as htmlmod
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))

CONST_FILES = [
    ("src/ai/reward/goal_shape_consts.rs", "goal"),
    ("src/ai/reward/dev_potential.rs", "dev"),
    ("src/ai/reward/economy_completion.rs", "econ"),
]

CONST_RE = re.compile(
    r"^pub const (?P<name>[A-Z0-9_]+): (?P<ty>[A-Za-z0-9_<>]+) = (?P<val>[^;]+);", re.M
)


def parse_consts(path, family):
    text = open(os.path.join(ROOT, path), encoding="utf-8").read()
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = CONST_RE.match(line)
        if not m:
            continue
        # Walk upward collecting a contiguous run of `///` doc lines.
        doc_lines = []
        j = i - 1
        while j >= 0 and lines[j].strip().startswith("///"):
            doc_lines.insert(0, lines[j].strip()[3:].strip())
            j -= 1
        doc = " ".join(doc_lines)
        out.append(
            {
                "name": m.group("name"),
                "ty": m.group("ty"),
                "val": m.group("val").strip(),
                "doc": doc,
                "file": path,
                "line": i + 1,
                "family": family,
                "first_fit": "first fit" in doc.lower(),
                "measured": "measured" in doc.lower(),
            }
        )
    return out


def numeric(v):
    try:
        return float(v)
    except ValueError:
        return None


def git_info():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        return sha, branch
    except Exception:
        return "unknown", "unknown"


# ---- Hand-curated sections (not `pub const`-parseable, or already have a
# canonical source elsewhere -- see the module doc above for why). --------

CURATED_CLI = [
    ("--td-w", "0.7", "label", "Weight of the TD(λ) delta vs. the flat final-outcome tail in the value LABEL. No-op if --no-reward-shaping."),
    ("--td-lambda", "0.8", "label", "TD(λ) trace decay -- sets the credit window's center of mass to 1/(1-λ) turns."),
    ("--outcome-scale", "3.0", "label", "Scale on the relative final-outcome ratio before the [-1,1] clamp in the value label."),
    ("--label-rel-w", "0.4", "label", "Relative weight used ONLY for TD(λ) label windows; the in-tree backup keeps reward::REL_W."),
    ("--wl-labels", "off", "label", "±1 win/loss value labels from the adjudicated winner, replacing the score-delta label entirely."),
    ("--no-reward-shaping", "off (shaping ON by default)", "label", "Opt out of the whole TD(λ)+outcome blend, falling back to flat final-outcome-only labels."),
    ("--shape-w-label", "0", "label", "Weight on dev_potential Φ (T1, EXP_ELO_016) in TD-label snapshots. Off by default."),
    ("--shape-w-tree", "0", "tree (Gumbel only)", "Weight on dev_potential Φ in the Gumbel backend's in-tree edge rewards. Off by default."),
    ("--pursuit-w-label", "0", "label", "Weight on pursuit_potential Φ (T2, EXP_ELO_018) in TD-label snapshots, independent of --shape-w-label."),
    ("--pursuit-w-tree", "0", "tree (Gumbel only)", "Weight on pursuit_potential Φ in the Gumbel backend's in-tree edge rewards."),
    ("--goal-channels", "off", "features + search", "Drives the appended goal channels with the scripted goal-setter (orders + stance + star gate)."),
    ("--goal-w-tree", "0", "tree (macro-mcts)", "Weight on goal_potential (T3) in net seats' in-tree edge rewards. Requires --goal-channels. ⚠️ see Known traps."),
    ("--macro-lambda", "1.0", "tree (macro-mcts)", "λ on Δφ in the ONE real per-ply commit (rank_view). This is the weight that actually governs live play."),
    ("--macro-rollout-lambda", "= macro-lambda", "tree (macro-mcts)", "λ for the internal search tree's OWN turn rollouts -- up to macro_sims calls per real turn."),
    ("--macro-shape-w", "0", "tree (macro-mcts)", "Weight on potential-based edge shaping inside the macro-mcts tree itself (distinct from macro-lambda)."),
    ("--macro-root-prior-w", "0", "search prior", "Weight on the macro policy head's PUCT-style prior at the search root. Costs an eval-server call per turn when nonzero."),
    ("--dagger-alpha", "0", "policy label", "DAgger expert dose: blends Greedy's move-ranking into the POLICY target at net-seat decisions. Not a value/Φ term."),
]

CURATED_SCORING = [
    ("Step onto Ruin/Village (uncaptured)", "+43.0", "scoring.rs ~513-524", "Flat capture bonus. Gated on get_city_at(...).is_none() since Aug 2026 -- previously fired on ANY city tile including your own."),
    ("Step onto enemy city", "+50.0", "scoring.rs ~526-531", "Flat bonus, tile.owner not in {self, 0}."),
    ("Step, base score", "35.0", "scoring.rs ~499", "Flat floor every Step candidate starts from."),
    ("Step, center-of-map pull", "up to +6.0", "scoring.rs ~613-614", "(6 - Manhattan dist to center).max(0). The term Verdi asked to compare against the lighthouse pull."),
    ("Step, unrevealed-Lighthouse-corner pull", "+10 flat (closing) + up to +12 (decays 2.0/tile, 0 past 6 tiles)", "scoring.rs, nearest_unrevealed_lighthouse_corner", "Added same session as the turn-1 fix. The +10 flat component is the one non-decaying piece in that block -- flagged, not yet changed (Verdi's call)."),
    ("Step, capturable-village closing pull", "+20 flat (closing) + (18 - 4·d).max(0)", "scoring.rs ~582-587", "Dominant Step term once a village is in reach."),
    ("Step, frontier-resource pull", "up to +14 flat + (8 - 1.5·d).max(0), × regional openness", "scoring.rs ~588-607", "Fires only when no capturable village is in sight -- the 'border fruit' fog-frontier heuristic."),
    ("Step, regional openness / newly-revealed fog", "×6.0 / ×4.0 (far) or ×2.0 / ×1.0 (approaching a village)", "scoring.rs ~536-577", "Damped near a village on purpose -- reveal-chasing beat the closing gradient in 85% of measured d=2 episodes."),
    ("Build/Harvest, Monument least-disruptive penalty", "-15.0 per hub-worthy resource cluster (Metal/Crop/Forest ≥2 adjacent)", "scoring.rs, monument placement fix", "Added same session -- previously EVERY legal Monument tile scored bit-identical."),
]

LAYER_LABEL = {
    "goal": "T3 goal-priced Φ (goal_potential.rs, EXP_ELO_028) — live in-tree on every macro-mcts ply, real-trajectory unless noted",
    "dev": "T1/T2 dev + pursuit label Φ (dev_potential.rs, EXP_ELO_016/018) — label-side by default, in-tree ONLY under the Gumbel backend with --shape-w-tree/--pursuit-w-tree nonzero",
    "econ": "Shared helper constant (economy_completion.rs) — consumed by goal_potential.rs's STRANDED/COMPLETION terms",
}

KNOWN_TRAPS = [
    ("--goal-w-tree defaults to 0", "self_play and arena's CLI default is 0.0, but production training (run_training_loop.sh) sets it to 1 whenever GOAL_CHANNELS=1. Omit it from a manual self_play/arena invocation and the entire T3 in-tree pricing channel silently goes dark -- the run looks normal, the numbers are just wrong. (memory: goal-w-tree-harness-trap)"),
    ("SHAPE_GOAL_LIGHTHOUSE reads as 0 at ply-ranking, always", "discovery.rs's simulating branch deliberately leaves tile.explorers untouched (peek-prevention), so a simulated candidate never reads a corner as \"explored\" -- this term's condition can never go true inside rank_plies. It still fires for real, permanently, on the REAL trajectory once a corner is actually revealed and stays in every subsequent Φ evaluation -- it just contributes nothing to the MARGINAL Δφ any candidate move is ranked on. Found diagnosing the Aug 23 lighthouse-vs-center question."),
    ("SHAPE_UNIT_GOAL_PER_TILE / SHAPE_CITY_TRAIN_BLOCKED are real-trajectory only", "Both gate on unit_goals.is_some(), which is only ever Some on MacroMctsAgent's real per-ply commit -- every internal rollout, and any call through the plain goal_potential()/goal_potential_with_threats() wrappers (which hardcode None), never sees either term. A goal_potential_tests.rs fixture that calls goal_potential() directly and expects these to fire will silently read 0."),
    ("goal.save_target isn't captured by POLYFISH_PLY_TRACE", "reward_lab (and anything else reconstructing a MacroGoal from the trace) can only recover stance + orders, so the SAVE ramp term (SHAPE_GOAL_SAVE) always reads as inactive when replaying a historical ply this way, even on a ply where it was genuinely live."),
]


def fmt_val(v, ty):
    n = numeric(v)
    if n is not None and ty in ("f32", "i32", "usize"):
        if n == int(n):
            return f"{int(n):,}"
        return f"{n:g}"
    return htmlmod.escape(v)


def render_table(rows, show_family_col=False):
    out = ['<table><thead><tr>']
    out.append('<th>Constant</th><th class="num">Value</th><th>Location</th><th>Notes</th><th>Evidence</th>')
    out.append('</tr></thead><tbody>')
    for r in rows:
        tags = []
        if r["first_fit"]:
            tags.append('<span class="tag tag-firstfit">first fit</span>')
        if r["measured"]:
            tags.append('<span class="tag tag-measured">measured</span>')
        if not tags:
            tags.append('<span class="tag tag-unknown">unmarked</span>')
        out.append(
            f'<tr id="c-{r["name"]}">'
            f'<td class="name"><code>{r["name"]}</code></td>'
            f'<td class="num">{fmt_val(r["val"], r["ty"])}</td>'
            f'<td class="loc"><code>{r["file"].split("/")[-1]}:{r["line"]}</code></td>'
            f'<td class="doc">{htmlmod.escape(r["doc"])}</td>'
            f'<td class="tags">{"".join(tags)}</td>'
            f'</tr>'
        )
    out.append('</tbody></table>')
    return "".join(out)


def render_ladder(all_rows):
    # f32 only -- i32/usize consts in these files are caps/ranges (tile
    # counts, corner counts), a different unit than score-equivalents, and
    # mixing them onto the same scale would be apples-to-oranges.
    scored = [r for r in all_rows if r["ty"] == "f32"]
    vals = sorted({numeric(r["val"]) for r in scored if numeric(r["val"]) is not None and numeric(r["val"]) > 0})
    if not vals:
        return ""
    vmax = max(vals)
    marks = []
    for v in vals:
        names = sorted({r["name"] for r in scored if numeric(r["val"]) == v})
        left = 4 + 92 * (v / vmax)
        title = ", ".join(names)
        marks.append(
            f'<div class="ladder-mark" style="left:{left:.2f}%" title="{htmlmod.escape(title)}">'
            f'<div class="ladder-tick"></div><div class="ladder-val">{fmt_val(str(v), "f32")}</div></div>'
        )
    return f'<div class="ladder"><div class="ladder-track"></div>{"".join(marks)}</div>'


def main():
    all_rows = []
    for path, family in CONST_FILES:
        all_rows.extend(parse_consts(path, family))
    all_rows.sort(key=lambda r: (numeric(r["val"]) is None, -(numeric(r["val"]) or 0)))

    sha, branch = git_info()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    goal_rows = [r for r in all_rows if r["family"] == "goal"]
    dev_rows = [r for r in all_rows if r["family"] == "dev"]
    econ_rows = [r for r in all_rows if r["family"] == "econ"]

    cli_rows = "".join(
        f'<tr><td class="name"><code>{htmlmod.escape(flag)}</code></td>'
        f'<td class="num">{htmlmod.escape(default)}</td>'
        f'<td class="loc">{htmlmod.escape(layer)}</td>'
        f'<td class="doc">{htmlmod.escape(desc)}</td></tr>'
        for flag, default, layer, desc in CURATED_CLI
    )
    scoring_rows = "".join(
        f'<tr><td class="name">{htmlmod.escape(name)}</td>'
        f'<td class="num">{htmlmod.escape(val)}</td>'
        f'<td class="loc"><code>{htmlmod.escape(loc)}</code></td>'
        f'<td class="doc">{htmlmod.escape(desc)}</td></tr>'
        for name, val, loc, desc in CURATED_SCORING
    )
    traps = "".join(
        f'<div class="trap"><div class="trap-title">⚠ {htmlmod.escape(title)}</div>'
        f'<div class="trap-body">{htmlmod.escape(body)}</div></div>'
        for title, body in KNOWN_TRAPS
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Polyfish Reward Catalog</title>
<style>
:root {{
  --bg: #fbfaf7; --surface: #ffffff; --border: #e4e0d8; --border-soft: #eeece5;
  --text: #221f1a; --text-dim: #6b6559; --text-faint: #8f897c;
  --accent: #b5502f; --accent-soft: #f4e3da;
  --mono: "SF Mono", ui-monospace, "JetBrains Mono", Menlo, monospace;
  --sans: -apple-system, "Inter", "Helvetica Neue", Arial, sans-serif;
  --tag-firstfit-bg: #f4e3da; --tag-firstfit-fg: #93401f;
  --tag-measured-bg: #e2ebe0; --tag-measured-fg: #3f6b39;
  --tag-unknown-bg: #ece9e2; --tag-unknown-fg: #7a7568;
  --trap-bg: #fbeee9; --trap-border: #dba48c;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #171512; --surface: #201d19; --border: #38332c; --border-soft: #2a2620;
    --text: #ede8e0; --text-dim: #a39c8c; --text-faint: #7a7364;
    --accent: #e08a5f; --accent-soft: #3a281f;
    --tag-firstfit-bg: #3a281f; --tag-firstfit-fg: #e5a685;
    --tag-measured-bg: #24312090; --tag-measured-fg: #9bc492;
    --tag-unknown-bg: #2c2822; --tag-unknown-fg: #a39c8c;
    --trap-bg: #2e1f1a; --trap-border: #6b3c2b;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #171512; --surface: #201d19; --border: #38332c; --border-soft: #2a2620;
  --text: #ede8e0; --text-dim: #a39c8c; --text-faint: #7a7364;
  --accent: #e08a5f; --accent-soft: #3a281f;
  --tag-firstfit-bg: #3a281f; --tag-firstfit-fg: #e5a685;
  --tag-measured-bg: #24312090; --tag-measured-fg: #9bc492;
  --tag-unknown-bg: #2c2822; --tag-unknown-fg: #a39c8c;
  --trap-bg: #2e1f1a; --trap-border: #6b3c2b;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text); font-family: var(--sans);
  margin: 0; padding: 0 0 6rem; line-height: 1.5;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; padding: 3rem 2rem 0; }}
header h1 {{
  font-size: 2rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; text-wrap: balance;
}}
header .sub {{ color: var(--text-dim); font-size: 0.98rem; max-width: 68ch; }}
header .meta {{
  margin-top: 1rem; font-family: var(--mono); font-size: 0.78rem; color: var(--text-faint);
  display: flex; gap: 1.2rem; flex-wrap: wrap;
}}
nav.toc {{
  display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.6rem 0 0; padding: 0;
}}
nav.toc a {{
  font-size: 0.82rem; color: var(--text-dim); text-decoration: none;
  border: 1px solid var(--border); border-radius: 999px; padding: 0.3rem 0.8rem;
}}
nav.toc a:hover {{ color: var(--accent); border-color: var(--accent); }}
section {{ margin-top: 3rem; }}
h2 {{
  font-size: 1.15rem; margin: 0 0 0.3rem; padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}}
h2 .count {{ color: var(--text-faint); font-weight: 400; font-size: 0.85em; }}
.layer-desc {{ color: var(--text-dim); font-size: 0.88rem; margin: 0.6rem 0 1.1rem; max-width: 78ch; }}
table {{
  width: 100%; border-collapse: collapse; font-size: 0.86rem;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden;
}}
.table-scroll {{ overflow-x: auto; border-radius: 10px; }}
th {{
  text-align: left; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.04em;
  text-transform: uppercase; color: var(--text-faint); padding: 0.6rem 0.9rem;
  border-bottom: 1px solid var(--border); background: var(--border-soft);
  white-space: nowrap;
}}
td {{ padding: 0.65rem 0.9rem; border-bottom: 1px solid var(--border-soft); vertical-align: top; }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: var(--border-soft); }}
td.name code, td.loc code {{ font-family: var(--mono); font-size: 0.82rem; }}
td.name code {{ color: var(--accent); }}
td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
td.doc {{ color: var(--text-dim); max-width: 46ch; }}
td.loc {{ color: var(--text-faint); white-space: nowrap; }}
.tag {{
  display: inline-block; font-size: 0.68rem; padding: 0.15rem 0.5rem; border-radius: 999px;
  font-weight: 600; letter-spacing: 0.01em; white-space: nowrap;
}}
.tag-firstfit {{ background: var(--tag-firstfit-bg); color: var(--tag-firstfit-fg); }}
.tag-measured {{ background: var(--tag-measured-bg); color: var(--tag-measured-fg); }}
.tag-unknown {{ background: var(--tag-unknown-bg); color: var(--tag-unknown-fg); }}
.ladder {{ position: relative; height: 4.4rem; margin: 1.6rem 0 2.4rem; }}
.ladder-track {{
  position: absolute; top: 2rem; left: 4%; right: 4%; height: 2px; background: var(--border);
}}
.ladder-mark {{ position: absolute; top: 0; transform: translateX(-50%); text-align: center; width: 6rem; }}
.ladder-tick {{
  width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
  margin: 1.68rem auto 0.4rem; box-shadow: 0 0 0 3px var(--bg);
}}
.ladder-val {{ font-family: var(--mono); font-size: 0.72rem; color: var(--text-dim); }}
.trap {{
  background: var(--trap-bg); border: 1px solid var(--trap-border); border-radius: 10px;
  padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
}}
.trap-title {{ font-weight: 600; font-size: 0.88rem; margin-bottom: 0.3rem; }}
.trap-body {{ font-size: 0.85rem; color: var(--text-dim); }}
.callout {{
  border-left: 3px solid var(--accent); background: var(--accent-soft);
  padding: 0.9rem 1.1rem; border-radius: 0 8px 8px 0; font-size: 0.88rem; color: var(--text-dim);
  margin: 1rem 0;
}}
.callout code {{ font-family: var(--mono); color: var(--text); }}
footer {{
  margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
  font-size: 0.8rem; color: var(--text-faint);
}}
footer code {{ font-family: var(--mono); background: var(--border-soft); padding: 0.1rem 0.4rem; border-radius: 4px; }}
input#filter {{
  font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.8rem; width: 100%;
  max-width: 320px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface); color: var(--text); margin-bottom: 1rem;
}}
input#filter:focus {{ outline: none; border-color: var(--accent); }}
</style></head>
<body>
<div class="wrap">
<header>
  <h1>Reward Catalog</h1>
  <div class="sub">Every named reward-shaping magnitude in Polyfish, generated straight from source so this page can never drift the way a hand-copied list would. Regenerate any time with <code>python3 gen_reward_catalog.py</code>.</div>
  <div class="meta">
    <span>commit <code>{sha}</code> ({branch})</span>
    <span>generated {generated}</span>
  </div>
  <nav class="toc">
    <a href="#ladder">Magnitude ladder</a>
    <a href="#goal">T3 goal-priced Φ</a>
    <a href="#dev">T1/T2 dev + pursuit Φ</a>
    <a href="#cli">CLI label/tree knobs</a>
    <a href="#scoring">Move-ranking heuristics</a>
    <a href="#traps">Known traps</a>
    <a href="#loop">Fast tuning loop</a>
  </nav>
</header>

<input id="filter" type="text" placeholder="Filter by name or text&hellip;" oninput="filterRows(this.value)">

<section id="ladder">
  <h2>Magnitude ladder</h2>
  <div class="layer-desc">Every distinct positive constant value across the goal-priced and dev/pursuit families, on one scale. Hover a mark for the constant name(s) at that value. A term you're tuning should land somewhere deliberate on this line, not off to one side by accident.</div>
  {render_ladder(goal_rows + dev_rows + econ_rows)}
</section>

<section id="goal">
  <h2>T3 goal-priced Φ <span class="count">({len(goal_rows)} constants, goal_shape_consts.rs)</span></h2>
  <div class="layer-desc">{LAYER_LABEL["goal"]}. This is the system every recent session's reward work has touched -- <code>goal_potential_breakdown()</code> (new) reports every one of these by name; see <a href="#loop">the fast tuning loop</a> below.</div>
  <div class="table-scroll">{render_table(goal_rows)}</div>
</section>

<section id="dev">
  <h2>T1/T2 dev + pursuit label Φ <span class="count">({len(dev_rows)} constants, dev_potential.rs)</span></h2>
  <div class="layer-desc">{LAYER_LABEL["dev"]}. Older (EXP_ELO_016/018) than the goal-priced system and off by default in current production training -- present for the Gumbel backend's label shaping, not macro-mcts.</div>
  <div class="table-scroll">{render_table(dev_rows)}</div>
</section>

<section id="econ">
  <h2>Shared helper constants <span class="count">({len(econ_rows)}, economy_completion.rs)</span></h2>
  <div class="table-scroll">{render_table(econ_rows)}</div>
</section>

<section id="cli">
  <h2>CLI label / in-tree weighting knobs <span class="count">(self_play.rs, hand-curated)</span></h2>
  <div class="layer-desc">Not <code>pub const</code> -- these are <code>self_play</code>/<code>arena</code> CLI flags that scale the Φ families above, or construct the value-target label directly. Canonical source of truth is always <code>self_play --help</code>; this table is a summary, kept intentionally short so it can't drift far from it.</div>
  <div class="table-scroll"><table><thead><tr><th>Flag</th><th class="num">Default</th><th>Layer</th><th>What it does</th></tr></thead><tbody>{cli_rows}</tbody></table></div>
</section>

<section id="scoring">
  <h2>Move-ranking heuristics <span class="count">(scoring.rs, hand-curated, representative)</span></h2>
  <div class="layer-desc">A DIFFERENT layer from everything above: <code>score_move()</code> ranks candidates BEFORE Δφ is added (<code>rank_plies</code> computes <code>score_move + λ·Δφ</code>). These are inline magic numbers, not named constants -- flagged here as un-hoisted rather than extracted exhaustively; this is a representative sample of the highest-impact ones, not the full file.</div>
  <div class="table-scroll"><table><thead><tr><th>Term</th><th class="num">Value</th><th>Location</th><th>Notes</th></tr></thead><tbody>{scoring_rows}</tbody></table></div>
</section>

<section>
  <h2>Named, not yet extracted</h2>
  <div class="callout">
    <strong>evaluator/*.rs</strong> (<code>evaluate_state</code>, the macro-leaf heuristic backend consulted when <code>--macro-leaf heuristic</code>) is a FOURTH reward-adjacent layer -- the fallback board evaluator macro-mcts uses instead of the network. It has its own weighting scheme, split by concern (<code>economy.rs</code>, <code>army.rs</code>, <code>research.rs</code>, <code>exploration.rs</code>, <code>gamestate.rs</code>). Out of scope for this generator; see <code>notes-heuristics.md</code> for its design spec.
  </div>
</section>

<section id="traps">
  <h2>Known traps</h2>
  {traps}
</section>

<section id="loop">
  <h2>Fast tuning loop</h2>
  <div class="layer-desc">
    Two loops, don't confuse them:
  </div>
  <div class="callout">
    <strong>Inner loop (seconds) -- does the arithmetic do what I intended?</strong><br>
    Edit a constant above &rarr; <code>cargo build --bin reward_lab</code> (debug, ~2-3s incremental) &rarr;
    <code>reward_lab --replay &lt;file&gt; --trace &lt;file&gt; --turn N --player P</code> against a frozen historical ply (&lt;1s) &rarr;
    read the per-term Δφ breakdown for every candidate. No self-play run, no eval server, no release rebuild.
    Reuses the SAME replay+trace pair across as many edits as you want. This is what changed the turn-1
    capital-block hunt from a 15-minute, one-off <code>POLYFISH_DPHI_PROBE</code> rebuild into a &lt;4s loop.
  </div>
  <div class="callout">
    <strong>Outer loop (minutes) -- does it actually help?</strong><br>
    A frozen-state term diff can never answer this -- self-play isn't move-for-move reproducible even at a
    fixed seed, so "generate a game and see how it changes" is a behavior verdict, not a diff. Use the
    frozen paired-seed gauge (seed 770425 harness, n=128, ~5 min, 0.078 win-rate noise floor) once the inner
    loop confirms the term does what you meant.
  </div>
</section>

<footer>
  Generated by <code>polyfish-rs/gen_reward_catalog.py</code> at commit <code>{sha}</code>. Every number above is parsed from the same source files the engine compiles -- if this page and the code ever disagree, the page is stale; rerun the generator.
</footer>
</div>
<script>
function filterRows(q) {{
  q = q.trim().toLowerCase();
  document.querySelectorAll('table tbody tr').forEach(function(tr) {{
    tr.style.display = (!q || tr.textContent.toLowerCase().includes(q)) ? '' : 'none';
  }});
}}
</script>
</body></html>"""
    print(html)


if __name__ == "__main__":
    main()
