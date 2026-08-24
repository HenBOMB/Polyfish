#!/usr/bin/env python3
"""Regenerates the reward catalog artifact from source, not from memory.

Scope: macro-mcts + the T3 executor ONLY. The older EXP_ELO_016/018
dev_potential.rs system (Gumbel backend's own in-tree edge reward, gated by
--shape-w-tree/--pursuit-w-tree, which only gumbel_mcts/rounds.rs reads) is
deliberately excluded -- Verdi's call, Aug 24 2026: this page tracks the
system actually in use, not every reward experiment the codebase has ever
carried. The one exception is SHAPE_PROX_CAP, physically defined in
dev_potential.rs but genuinely consumed by goal_potential.rs's own EXPAND/
UNIT_GOAL proximity terms -- dropping it would misrepresent T3 itself, so
it's pulled in and filed under the goal-priced section, not the excluded one.

Parses every `pub const` (with its preceding `///` doc comment) out of the
files that actually define reward-shaping magnitudes -- goal_shape_consts.rs
(the T3 goal-priced Phi) and economy_completion.rs (a shared helper constant
goal_potential.rs imports). Everything else on the page (scoring.rs's inline
heuristic weights, the self_play CLI label/tree knobs, the evaluator/* leaf
heuristic) is hand-curated and clearly marked as such -- see CURATED_* below
-- because it either isn't a `pub const` at a stable location (scoring.rs)
or already has a canonical source of truth that would drift if duplicated
here (`self_play --help`).

Category (explore/economy/military) and carrot-vs-stick tags are also
hand-curated -- domain judgment, not mechanically derivable for most
constants. Carrot/stick IS grounded in source where possible: every T3 term
routes through PhiAcc::add or PhiAcc::sub in goal_potential.rs, and the
`sub` call sites (connect, city_risk, city_train_blocked, stranded) are
exactly the four stick terms below -- see STICK_LABELS. Constants that are
multipliers/caps INSIDE an add-term (e.g. a damping factor, a proximity cap)
inherit that parent term's polarity rather than getting their own sign.

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
    ("src/ai/reward/economy_completion.rs", "econ"),
]

# Grepped directly from goal_potential.rs: the only `acc.sub(...)` labels.
# Every other term is `acc.add(...)`. See the module doc above.
STICK_LABELS = {"connect", "city_risk", "city_train_blocked", "stranded"}

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


# ---- Category + polarity tags (hand-curated, see module doc). -----------
# (categories..., "carrot"|"stick")
TAGS = {
    "SHAPE_GOAL_SPT": (("economy",), "carrot"),
    "SHAPE_GOAL_ARM_PER_COST": (("military",), "carrot"),
    "SHAPE_GOAL_ARM_SPT": (("economy",), "carrot"),
    "SHAPE_GOAL_EXPAND_PER_TILE": (("explore",), "carrot"),
    "SHAPE_GOAL_TECH_FIT": (("economy",), "carrot"),
    "SHAPE_GOAL_CONNECT": (("economy",), "stick"),
    "SHAPE_GOAL_SCOUT": (("explore",), "carrot"),
    "SHAPE_GOAL_EXPAND_DONE": (("explore",), "carrot"),
    "SHAPE_UNIT_GOAL_PER_TILE": (("explore",), "carrot"),
    "SHAPE_UNIT_GOAL_COMPLETE": (("explore",), "carrot"),
    "SHAPE_GOAL_RIDER": (("military",), "carrot"),
    "SHAPE_GOAL_LANE_PER_COST": (("military",), "carrot"),
    "SCOUT_QUADRANT_CAP": (("explore",), "carrot"),
    "SHAPE_GOAL_LIGHTHOUSE": (("explore",), "carrot"),
    "SHAPE_GOAL_EXPLORER": (("explore",), "carrot"),
    "SHAPE_GOAL_EXPLORER_LIGHTHOUSE": (("explore",), "carrot"),
    "EXPLORER_WALK_RANGE": (("explore",), "carrot"),
    "EXPLORER_CORNER_CAP": (("explore",), "carrot"),
    "SHAPE_GOAL_YIELD_ADJ": (("economy",), "carrot"),
    "SHAPE_GOAL_YIELD_CAPACITY_W": (("economy",), "carrot"),
    "SHAPE_GOAL_YIELD_ADJ_STARS": (("economy",), "carrot"),
    "SHAPE_GOAL_FOREST_STANDING": (("economy",), "carrot"),
    "SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE": (("explore",), "carrot"),
    "SHAPE_GOAL_STRANDED": (("economy",), "stick"),
    "SHAPE_GOAL_SAVE": (("economy",), "carrot"),
    "SHAPE_GOAL_SUPER": (("military",), "carrot"),
    "SHAPE_GOAL_CITY_RISK": (("military",), "stick"),
    "SHAPE_CITY_TRAIN_BLOCKED": (("economy",), "stick"),
    "SHAPE_GOAL_DEFEND_COVER": (("military",), "carrot"),
    "SHAPE_GOAL_DEFEND_HOLD": (("military",), "carrot"),
    "SHAPE_GOAL_ATTACK_PRESS": (("military",), "carrot"),
    "SHAPE_GOAL_SIEGE_HOLD_MULT": (("military",), "carrot"),
    "SHAPE_GOAL_SUPER_ECON_DAMP": (("military",), "carrot"),
    "SHAPE_GOAL_COMPLETION": (("economy",), "carrot"),
    "SHAPE_GOAL_RETAKE_W": (("explore", "military"), "carrot"),
    "SHAPE_GOAL_RUIN_W": (("explore",), "carrot"),
    "SHAPE_GOAL_CONTEST_SECOND": (("explore", "military"), "carrot"),
    "SHAPE_GOAL_BODY": (("explore", "military"), "carrot"),
    "BODY_CAP_MAX": (("explore", "military"), "carrot"),
    "STRANDED_PER_CITY_CAP": (("economy",), "stick"),
    "SHAPE_PROX_CAP": (("explore",), "carrot"),
}

# ---- Hand-curated sections (not `pub const`-parseable, or already have a
# canonical source elsewhere -- see the module doc above for why). --------
# T1/T2 label-only knobs (--shape-w-*, --pursuit-w-*) are intentionally
# absent: they gate dev_potential.rs, out of scope per the module doc.

CURATED_CLI = [
    ("--td-w", "0.7", "label", "Weight of the TD(λ) delta vs. the flat final-outcome tail in the value LABEL. No-op if --no-reward-shaping."),
    ("--td-lambda", "0.8", "label", "TD(λ) trace decay -- sets the credit window's center of mass to 1/(1-λ) turns."),
    ("--outcome-scale", "3.0", "label", "Scale on the relative final-outcome ratio before the [-1,1] clamp in the value label."),
    ("--label-rel-w", "0.4", "label", "Relative weight used ONLY for TD(λ) label windows; the in-tree backup keeps reward::REL_W."),
    ("--wl-labels", "off", "label", "±1 win/loss value labels from the adjudicated winner, replacing the score-delta label entirely."),
    ("--no-reward-shaping", "off (shaping ON by default)", "label", "Opt out of the whole TD(λ)+outcome blend, falling back to flat final-outcome-only labels."),
    ("--goal-channels", "off", "features + search", "Drives the appended goal channels with the scripted goal-setter (orders + stance + star gate)."),
    ("--goal-w-tree", "0", "tree (macro-mcts)", "Weight on goal_potential (T3) in net seats' in-tree edge rewards. Requires --goal-channels. ⚠️ see Known traps."),
    ("--macro-lambda", "1.0", "tree (macro-mcts)", "λ on Δφ in the ONE real per-ply commit (rank_view). This is the weight that actually governs live play."),
    ("--macro-rollout-lambda", "= macro-lambda", "tree (macro-mcts)", "λ for the internal search tree's OWN turn rollouts -- up to macro_sims calls per real turn."),
    ("--macro-shape-w", "0", "tree (macro-mcts)", "Weight on potential-based edge shaping inside the macro-mcts tree itself (distinct from macro-lambda)."),
    ("--macro-root-prior-w", "0", "search prior", "Weight on the macro policy head's PUCT-style prior at the search root. Costs an eval-server call per turn when nonzero."),
    ("--dagger-alpha", "0", "policy label", "DAgger expert dose: blends Greedy's move-ranking into the POLICY target at net-seat decisions. Not a value/Φ term. Backend-agnostic (checks is_net_seat, not search backend)."),
]

# (name, value, location, notes, categories, polarity)
CURATED_SCORING = [
    ("Step onto Ruin/Village (uncaptured)", "+43.0", "scoring.rs ~513-524", "Flat capture bonus. Gated on get_city_at(...).is_none() since Aug 2026 -- previously fired on ANY city tile including your own.", ("explore",), "carrot"),
    ("Step onto enemy city", "+50.0", "scoring.rs ~526-531", "Flat bonus, tile.owner not in {self, 0}.", ("military",), "carrot"),
    ("Step, base score", "35.0", "scoring.rs ~499", "Flat floor every Step candidate starts from.", ("explore",), "carrot"),
    ("Step, center-of-map pull", "up to +6.0", "scoring.rs ~613-614", "(6 - Manhattan dist to center).max(0). The term Verdi asked to compare against the lighthouse pull.", ("explore",), "carrot"),
    ("Step, unrevealed-Lighthouse-corner pull", "+10 flat (closing) + up to +12 (decays 2.0/tile, 0 past 6 tiles)", "scoring.rs, nearest_unrevealed_lighthouse_corner", "Added same session as the turn-1 fix. The +10 flat component is the one non-decaying piece in that block -- flagged, not yet changed (Verdi's call).", ("explore",), "carrot"),
    ("Step, capturable-village closing pull", "+20 flat (closing) + (18 - 4·d).max(0)", "scoring.rs ~582-587", "Dominant Step term once a village is in reach.", ("explore",), "carrot"),
    ("Step, frontier-resource pull", "up to +14 flat + (8 - 1.5·d).max(0), × regional openness", "scoring.rs ~588-607", "Fires only when no capturable village is in sight -- the 'border fruit' fog-frontier heuristic.", ("explore", "economy"), "carrot"),
    ("Step, regional openness / newly-revealed fog", "×6.0 / ×4.0 (far) or ×2.0 / ×1.0 (approaching a village)", "scoring.rs ~536-577", "Damped near a village on purpose -- reveal-chasing beat the closing gradient in 85% of measured d=2 episodes.", ("explore",), "carrot"),
    ("Build/Harvest, Monument least-disruptive penalty", "-15.0 per hub-worthy resource cluster (Metal/Crop/Forest ≥2 adjacent)", "scoring.rs, monument placement fix", "Added same session -- previously EVERY legal Monument tile scored bit-identical.", ("economy",), "stick"),
]

LAYER_LABEL = {
    "goal": "T3 goal-priced Φ (goal_potential.rs, EXP_ELO_028) — the macro-mcts executor's live in-tree Φ, real-trajectory unless noted",
    "econ": "Shared helper constant (economy_completion.rs) — consumed by goal_potential.rs's STRANDED/COMPLETION terms",
}

CAT_LABEL = {"explore": "Explore", "economy": "Economy", "military": "Military"}
POL_LABEL = {"carrot": "Carrot", "stick": "Stick"}

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


def tag_chips(cats, polarity):
    chips = [f'<span class="tag tag-cat-{c}">{CAT_LABEL[c]}</span>' for c in cats]
    chips.append(f'<span class="tag tag-pol-{polarity}">{POL_LABEL[polarity]}</span>')
    return "".join(chips)


def evidence_chips(r):
    tags = []
    if r["first_fit"]:
        tags.append('<span class="tag tag-firstfit">first fit</span>')
    if r["measured"]:
        tags.append('<span class="tag tag-measured">measured</span>')
    if not tags:
        tags.append('<span class="tag tag-unknown">unmarked</span>')
    return "".join(tags)


def render_table(rows):
    out = ['<table><thead><tr>']
    out.append(
        '<th>Constant</th><th class="num">Value</th><th>Location</th><th>Notes</th>'
        '<th>Tags</th><th>Evidence</th>'
    )
    out.append('</tr></thead><tbody>')
    for r in rows:
        cats, polarity = TAGS.get(r["name"], ((), None))
        cat_str = " ".join(cats)
        out.append(
            f'<tr id="c-{r["name"]}" data-cats="{cat_str}" data-pol="{polarity or ""}">'
            f'<td class="name"><code>{r["name"]}</code></td>'
            f'<td class="num">{fmt_val(r["val"], r["ty"])}</td>'
            f'<td class="loc"><code>{r["file"].split("/")[-1]}:{r["line"]}</code></td>'
            f'<td class="doc">{htmlmod.escape(r["doc"])}</td>'
            f'<td class="tags">{tag_chips(cats, polarity) if polarity else ""}</td>'
            f'<td class="tags">{evidence_chips(r)}</td>'
            f'</tr>'
        )
    out.append('</tbody></table>')
    return "".join(out)


def render_ladder(all_rows):
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
    # SHAPE_PROX_CAP: physically in dev_potential.rs, genuinely consumed by
    # goal_potential.rs. Pull it in, file it under "goal". See module doc.
    dev_rows = parse_consts("src/ai/reward/dev_potential.rs", "goal")
    all_rows.extend(r for r in dev_rows if r["name"] == "SHAPE_PROX_CAP")

    all_rows.sort(key=lambda r: (numeric(r["val"]) is None, -(numeric(r["val"]) or 0)))

    sha, branch = git_info()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    goal_rows = [r for r in all_rows if r["family"] == "goal"]
    econ_rows = [r for r in all_rows if r["family"] == "econ"]

    cli_rows = "".join(
        f'<tr><td class="name"><code>{htmlmod.escape(flag)}</code></td>'
        f'<td class="num">{htmlmod.escape(default)}</td>'
        f'<td class="loc">{htmlmod.escape(layer)}</td>'
        f'<td class="doc">{htmlmod.escape(desc)}</td></tr>'
        for flag, default, layer, desc in CURATED_CLI
    )
    scoring_rows = "".join(
        f'<tr data-cats="{" ".join(cats)}" data-pol="{polarity}">'
        f'<td class="name">{htmlmod.escape(name)}</td>'
        f'<td class="num">{htmlmod.escape(val)}</td>'
        f'<td class="loc"><code>{htmlmod.escape(loc)}</code></td>'
        f'<td class="doc">{htmlmod.escape(desc)}</td>'
        f'<td class="tags">{tag_chips(cats, polarity)}</td></tr>'
        for name, val, loc, desc, cats, polarity in CURATED_SCORING
    )
    traps = "".join(
        f'<div class="trap"><div class="trap-title">⚠ {htmlmod.escape(title)}</div>'
        f'<div class="trap-body">{htmlmod.escape(body)}</div></div>'
        for title, body in KNOWN_TRAPS
    )

    cat_chip_buttons = "".join(
        f'<button class="chip chip-cat-{c}" data-cat="{c}" onclick="toggleChip(this)">{CAT_LABEL[c]}</button>'
        for c in ("explore", "economy", "military")
    )
    pol_chip_buttons = "".join(
        f'<button class="chip chip-pol-{p}" data-pol="{p}" onclick="toggleChip(this)">{POL_LABEL[p]}</button>'
        for p in ("carrot", "stick")
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
  --cat-explore-bg: #e4ecf5; --cat-explore-fg: #2f5788;
  --cat-economy-bg: #eef0dd; --cat-economy-fg: #5c6b1f;
  --cat-military-bg: #f5e2df; --cat-military-fg: #9c3f2e;
  --pol-carrot-bg: #e2ebe0; --pol-carrot-fg: #3f6b39;
  --pol-stick-bg: #f4e0e0; --pol-stick-fg: #96372f;
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
    --cat-explore-bg: #24344a; --cat-explore-fg: #9dbde5;
    --cat-economy-bg: #333a1c; --cat-economy-fg: #c3d17f;
    --cat-military-bg: #3d2620; --cat-military-fg: #e5a08c;
    --pol-carrot-bg: #243120; --pol-carrot-fg: #9bc492;
    --pol-stick-bg: #3a2220; --pol-stick-fg: #e29a90;
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
  --cat-explore-bg: #24344a; --cat-explore-fg: #9dbde5;
  --cat-economy-bg: #333a1c; --cat-economy-fg: #c3d17f;
  --cat-military-bg: #3d2620; --cat-military-fg: #e5a08c;
  --pol-carrot-bg: #243120; --pol-carrot-fg: #9bc492;
  --pol-stick-bg: #3a2220; --pol-stick-fg: #e29a90;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text); font-family: var(--sans);
  margin: 0; padding: 0 0 6rem; line-height: 1.5;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 3rem 2rem 0; }}
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
td.doc {{ color: var(--text-dim); max-width: 42ch; }}
td.loc {{ color: var(--text-faint); white-space: nowrap; }}
td.tags {{ white-space: nowrap; }}
.tag {{
  display: inline-block; font-size: 0.68rem; padding: 0.15rem 0.5rem; border-radius: 999px;
  font-weight: 600; letter-spacing: 0.01em; white-space: nowrap; margin: 0 0.2rem 0.2rem 0;
}}
.tag-firstfit {{ background: var(--tag-firstfit-bg); color: var(--tag-firstfit-fg); }}
.tag-measured {{ background: var(--tag-measured-bg); color: var(--tag-measured-fg); }}
.tag-unknown {{ background: var(--tag-unknown-bg); color: var(--tag-unknown-fg); }}
.tag-cat-explore {{ background: var(--cat-explore-bg); color: var(--cat-explore-fg); }}
.tag-cat-economy {{ background: var(--cat-economy-bg); color: var(--cat-economy-fg); }}
.tag-cat-military {{ background: var(--cat-military-bg); color: var(--cat-military-fg); }}
.tag-pol-carrot {{ background: var(--pol-carrot-bg); color: var(--pol-carrot-fg); }}
.tag-pol-stick {{ background: var(--pol-stick-bg); color: var(--pol-stick-fg); }}
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
.controls {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; align-items: center; margin-bottom: 1.6rem; }}
input#filter {{
  font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.8rem; width: 100%;
  max-width: 320px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface); color: var(--text);
}}
input#filter:focus {{ outline: none; border-color: var(--accent); }}
.chip-group {{ display: flex; gap: 0.4rem; align-items: center; }}
.chip-group-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-faint); margin-right: 0.2rem; }}
button.chip {{
  font-family: var(--sans); font-size: 0.78rem; font-weight: 600; cursor: pointer;
  border: 1px solid var(--border); background: var(--surface); color: var(--text-dim);
  border-radius: 999px; padding: 0.3rem 0.85rem; transition: all 0.1s;
}}
button.chip.active.chip-cat-explore {{ background: var(--cat-explore-bg); color: var(--cat-explore-fg); border-color: var(--cat-explore-fg); }}
button.chip.active.chip-cat-economy {{ background: var(--cat-economy-bg); color: var(--cat-economy-fg); border-color: var(--cat-economy-fg); }}
button.chip.active.chip-cat-military {{ background: var(--cat-military-bg); color: var(--cat-military-fg); border-color: var(--cat-military-fg); }}
button.chip.active.chip-pol-carrot {{ background: var(--pol-carrot-bg); color: var(--pol-carrot-fg); border-color: var(--pol-carrot-fg); }}
button.chip.active.chip-pol-stick {{ background: var(--pol-stick-bg); color: var(--pol-stick-fg); border-color: var(--pol-stick-fg); }}
button.chip-reset {{ font-size: 0.78rem; color: var(--text-faint); background: none; border: none; cursor: pointer; text-decoration: underline; }}
</style></head>
<body>
<div class="wrap">
<header>
  <h1>Reward Catalog</h1>
  <div class="sub">Every named reward-shaping magnitude actually in use by macro-mcts and the T3 executor, generated straight from source so this page can never drift the way a hand-copied list would. Tag each term by category (explore / economy / military) and by whether it's a carrot (positive incentive) or a stick (negative one). Regenerate any time with <code>python3 gen_reward_catalog.py</code>.</div>
  <div class="meta">
    <span>commit <code>{sha}</code> ({branch})</span>
    <span>generated {generated}</span>
  </div>
  <nav class="toc">
    <a href="#ladder">Magnitude ladder</a>
    <a href="#goal">T3 goal-priced Φ</a>
    <a href="#cli">CLI label/tree knobs</a>
    <a href="#scoring">Move-ranking heuristics</a>
    <a href="#traps">Known traps</a>
    <a href="#loop">Fast tuning loop</a>
  </nav>
</header>

<div class="controls">
  <input id="filter" type="text" placeholder="Filter by name or text&hellip;" oninput="applyFilters()">
  <div class="chip-group"><span class="chip-group-label">Category</span>{cat_chip_buttons}</div>
  <div class="chip-group"><span class="chip-group-label">Incentive</span>{pol_chip_buttons}</div>
  <button class="chip-reset" onclick="resetChips()">reset tags</button>
</div>

<section id="ladder">
  <h2>Magnitude ladder</h2>
  <div class="layer-desc">Every distinct positive T3 constant value, on one scale (score-equivalent units only -- caps and tile-ranges are a different unit and excluded). Hover a mark for the constant name(s) at that value. A term you're tuning should land somewhere deliberate on this line, not off to one side by accident.</div>
  {render_ladder(goal_rows + econ_rows)}
</section>

<section id="goal">
  <h2>T3 goal-priced Φ <span class="count">({len(goal_rows)} constants, goal_shape_consts.rs)</span></h2>
  <div class="layer-desc">{LAYER_LABEL["goal"]}. <code>goal_potential_breakdown()</code> (new) reports every one of these by name; see <a href="#loop">the fast tuning loop</a> below.</div>
  <div class="table-scroll">{render_table(goal_rows)}</div>
</section>

<section id="econ">
  <h2>Shared helper constants <span class="count">({len(econ_rows)}, economy_completion.rs)</span></h2>
  <div class="table-scroll">{render_table(econ_rows)}</div>
</section>

<section id="cli">
  <h2>CLI label / in-tree weighting knobs <span class="count">(self_play.rs, hand-curated)</span></h2>
  <div class="layer-desc">Not <code>pub const</code> -- these are <code>self_play</code>/<code>arena</code> CLI flags that scale the Φ family above, or construct the value-target label directly. Canonical source of truth is always <code>self_play --help</code>; this table is a summary, kept intentionally short so it can't drift far from it. The Gumbel-only <code>--shape-w-*</code>/<code>--pursuit-w-*</code> pair (dev_potential.rs's own in-tree/label knobs) is out of scope here -- see the module doc in <code>gen_reward_catalog.py</code>.</div>
  <div class="table-scroll"><table><thead><tr><th>Flag</th><th class="num">Default</th><th>Layer</th><th>What it does</th></tr></thead><tbody>{cli_rows}</tbody></table></div>
</section>

<section id="scoring">
  <h2>Move-ranking heuristics <span class="count">(scoring.rs, hand-curated, representative)</span></h2>
  <div class="layer-desc">A DIFFERENT layer from everything above: <code>score_move()</code> ranks candidates BEFORE Δφ is added (<code>rank_plies</code> computes <code>score_move + λ·Δφ</code>) -- and it's the function macro-mcts's real per-ply commit actually calls, not a Gumbel-only path. These are inline magic numbers, not named constants -- flagged here as un-hoisted rather than extracted exhaustively; this is a representative sample of the highest-impact ones, not the full file.</div>
  <div class="table-scroll"><table><thead><tr><th>Term</th><th class="num">Value</th><th>Location</th><th>Notes</th><th>Tags</th></tr></thead><tbody>{scoring_rows}</tbody></table></div>
</section>

<section>
  <h2>Named, not yet extracted</h2>
  <div class="callout">
    <strong>evaluator/*.rs</strong> (<code>evaluate_state</code>, the macro-leaf heuristic backend consulted when <code>--macro-leaf heuristic</code>) is a further reward-adjacent layer -- the fallback board evaluator macro-mcts uses instead of the network. It has its own weighting scheme, split by concern (<code>economy.rs</code>, <code>army.rs</code>, <code>research.rs</code>, <code>exploration.rs</code>, <code>gamestate.rs</code>). Out of scope for this generator; see <code>notes-heuristics.md</code> for its design spec.
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
  Generated by <code>polyfish-rs/gen_reward_catalog.py</code> at commit <code>{sha}</code>. Every number above is parsed from the same source files the engine compiles -- if this page and the code ever disagree, the page is stale; rerun the generator. Category and carrot/stick tags are hand-curated (see the script's module doc); everything else on this page is either generated or explicitly marked as curated.
</footer>
</div>
<script>
var activeCats = new Set();
var activePols = new Set();

function toggleChip(btn) {{
  var cat = btn.getAttribute('data-cat');
  var pol = btn.getAttribute('data-pol');
  if (cat) {{ activeCats.has(cat) ? activeCats.delete(cat) : activeCats.add(cat); }}
  if (pol) {{ activePols.has(pol) ? activePols.delete(pol) : activePols.add(pol); }}
  btn.classList.toggle('active');
  applyFilters();
}}

function resetChips() {{
  activeCats.clear();
  activePols.clear();
  document.querySelectorAll('button.chip').forEach(function(b) {{ b.classList.remove('active'); }});
  applyFilters();
}}

function applyFilters() {{
  var q = document.getElementById('filter').value.trim().toLowerCase();
  document.querySelectorAll('table tbody tr').forEach(function(tr) {{
    var textOk = !q || tr.textContent.toLowerCase().includes(q);
    var rowCats = (tr.getAttribute('data-cats') || '').split(' ').filter(Boolean);
    var rowPol = tr.getAttribute('data-pol') || '';
    var catOk = activeCats.size === 0 || rowCats.some(function(c) {{ return activeCats.has(c); }});
    var polOk = activePols.size === 0 || activePols.has(rowPol);
    // Rows with no tags at all (e.g. CLI knobs table) always pass the tag filters.
    var hasTagAttrs = tr.hasAttribute('data-cats');
    tr.style.display = (textOk && (!hasTagAttrs || (catOk && polOk))) ? '' : 'none';
  }});
}}
</script>
</body></html>"""
    print(html)


if __name__ == "__main__":
    main()
