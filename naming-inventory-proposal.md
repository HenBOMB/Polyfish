# Function naming inventory — a rename proposal

**STATUS: IMPLEMENTED, Aug 17 2026.** Every rename below shipped (commits
`60b648d`, `974e7d2`), Section C's "lane" call was made by Verdi and applied
(including `SaveLane` → `SaveTarget` to free up the word), and the HTML
architecture report was regenerated to match — see the artifact for the
current names/locations. This document is kept as the historical record of
the proposal and its rationale, not a to-do list.

Requested by Verdi: many `src/ai/` function names are unintuitive. This is an
inventory of the ones flagged plus a few more in the same neighborhood, each
with a proposed rename and a description in the `What` / `How` format
requested for the HTML architecture report's "Computes" column.

**Scope.** All eight explicitly-named examples live in one place: the T1/T2
macro-orchestration layer (`oracle_macro.rs` plus its Aug 2026 split-outs —
`search/archetype.rs`, `search/goal_aux.rs`, `economy.rs`). That's not a
coincidence — this layer is where the codebase's internal jargon
("stance", "star gate", "aux", "scripted") gets used as function-name
shorthand without ever being spelled out. Section A covers those eight.
Section B adds a few more from the same layer with the same problem, found
while reading through it for Section A. Section C is a discipline flag:
"playstyle" / "archetype" / "lane" are three different names for the *same*
concept, used interchangeably. Section D is housekeeping for the HTML
report itself — four functions it documents no longer exist under those
names, having already been merged/retired earlier this session.

**What this is not.** A proposal to rename the *types* these functions are
built around (`MacroGoal`, `GoalAux`, `Stance`, `OrderKind`) — some of the
function-name vagueness traces back to those, but renaming a public type
touches far more call sites than renaming a function, and wasn't what was
asked. Flagged as an optional follow-up in Section C, not undertaken here.

Nothing in this document has been applied to the codebase yet. It's a
proposal to react to, not a change already made.

---

## A. The eight flagged names

### 1. `scripted_goal` → `compute_macro_goal`
**File:** `oracle_macro.rs:65`

*Current problem:* "scripted" describes an implementation detail (hand-written
rules, as opposed to a future learned policy) — it says nothing about what
the function actually produces. A reader has to already know the codebase's
vocabulary to guess this is the core per-ply decision function.

- **What:** Decides this ply's `MacroGoal` — which villages/cities to target
  (Expand/Attack/Defend orders) and which global spending stance
  (Grow/Arm/Save) to hold.
- **How:** Paints an Expand order on every still-capturable village; adds
  Attack orders where local force has a 1.5× superiority margin; unions in
  Defend orders from `city_risks`; picks Save only if a reachable savings
  target exists, else falls back to Arm (if threatened or primed) or Grow.

*Why this name:* mirrors its return type (`MacroGoal`) directly — "what
computes a `MacroGoal`? `compute_macro_goal`."

### 2. `update_goal` → `commit_macro_goal`
**File:** `oracle_macro.rs:377`

*Current problem:* "update" is close to content-free — nearly any function
in a stateful system "updates" something. Doesn't hint that this is
specifically the hysteresis wrapper around #1, or that "goal" here means
the *committed*, sticky version rather than the raw per-ply read.

- **What:** The stance-holding wrapper around `compute_macro_goal` — orders
  pass through unchanged, but a stance swing must hold for
  `STANCE_SWITCH_TURNS` (2) turns before it's adopted, except a threat
  response, which always lands immediately.
- **How:** Calls `compute_macro_goal`, then diffs the fresh stance against
  the currently-held one; on a mismatch it either starts/extends a
  challenger streak or (if the streak is long enough, or the goal is
  urgent) promotes the challenger and records a flip.

*Why this name:* "commit" signals the hysteresis/stickiness that "update"
doesn't, and pairs cleanly with #1 — compute the raw read, then commit it.

### 3. `scripted_goal_aux` → `compute_goal_aux`
**File:** `search/goal_aux.rs:90`

*Current problem:* same "scripted" issue as #1, compounded by "aux" being
unexplained shorthand for a specific type (`GoalAux`) a reader hasn't
necessarily met yet.

- **What:** Builds `GoalAux`, the T2→T3 supporting-context bundle handed
  down alongside the goal — city risk assessments, the recommended tech
  list, the save-lane's next tech, road tiles still needed, preferred
  units, and several whole-game purchase counters.
- **How:** Mostly assembly, not computation — it calls `city_risks`,
  `recommended_techs`, `connect_remaining`, `stance_pressure` (see #5) and
  the active archetype's overlays, and packages their outputs into one
  struct so Tier 3 doesn't have to re-derive any of it per candidate move.

*Why this name:* same reasoning as #1 — mirrors the `GoalAux` return type.

### 4. `save_batch_plan` → `pick_save_lane`
**File:** `economy.rs:265`

*Current problem:* "batch" implies a collection of multiple things being
saved for, but the function picks exactly *one* hub (a `SaveLane`: one
structure + one tech + a cost) to bank toward. The name describes a
plan-shaped return value that isn't what's returned.

- **What:** Picks the single hub structure worth banking stars for right
  now, or `None` if nothing qualifies.
- **How:** Filters every hub lane to ones reachable within
  `SAVE_MAX_TURNS` of current income, ranks survivors by population (or
  star) yield per star on this specific map, and returns the best one —
  ties broken toward the cheaper option.

*Why this name:* "pick" signals selection-among-options (which is what it
does), and the return type is literally a `SaveLane` — the name should say
so.

### 5. `stance_strength` → `stance_pressure`
**File:** `oracle_macro.rs:208`

*Current problem:* "strength of a stance-strength" is circular — the word
doesn't explain itself. It's actually two continuous pressure readings
(how much this position wants ARM vs. GROW) plus the reason for the
dominant one.

- **What:** Continuous 0–1 pressure toward ARM and toward GROW (unlike the
  categorical `Stance` enum, which only ever picks one), plus whether the
  ARM pressure comes from a threat or from momentum.
- **How:** Threat pressure is contested-cities-fraction weighted by local
  force ratio; momentum pressure is army-share advantage weighted by
  attackable-target share; GROW pressure is buyable population plus open
  expansion targets, normalized against fixed caps.

*Why this name:* "pressure" is a more specific, less circular word than
"strength" for "how much this pulls the stance in one direction."

### 6. `passes_star_gate` → `passes_stance_tech_mask`
**File:** `search/goal_aux.rs:466`

*Current problem:* "star gate" is unexplained internal jargon — it has
nothing to do with stars passing through anything; it's a per-move check
of whether a Research move's tech CLASS (pure-combat vs. pure-eco) matches
what the current stance permits.

- **What:** Whether one specific Research move is allowed under the
  current stance's tech-class rule. Non-Research moves always pass.
- **How:** GROW/SAVE drop pure-combat tech; ARM drops pure-eco tech, but
  only once ARM pressure (see #5) is near-certain (≥0.98) — a covered
  skirmish must not lock out the economy. Dual-class tech (fields a unit
  *and* opens a yielding structure) is never dropped by either rule.

*Why this name:* replaces the unexplained "star gate" with what the check
actually is — a stance-conditioned tech-class mask.

### 7. `passes_tech_caps` → `passes_tech_purchase_limits`
**File:** `search/goal_aux.rs:280`

*Current problem:* "caps" undersells what the function checks — alongside
the two whole-game purchase COUNTS (8 techs, 2 tier-3s) it also enforces
lane discipline (only the committed lane's/save-plan's own next tech may
be bought), the dry-map water-tech mask, and the FreeSpirit/Chivalry
stepping-stone rule. A reader expecting only a count check will be
surprised by the other three.

- **What:** Whether a Research move is allowed by any of four rules:
  whole-game purchase counts, the committed lane/save-plan whitelist, the
  no-water dead-end mask, and the knight stepping-stone rule.
- **How:** Each rule is checked independently and short-circuits to
  `false` on the first violation; a non-Research move always passes.

*Why this name:* "purchase limits" is closer to the true scope than
"caps" while staying short; the doc comment (not the name) is where the
other three rules get spelled out in full.

### 8. `goal_star_gate` → `tech_discipline_active`
**File:** `oracle_macro.rs:449`

*Current problem:* the single worst offender in this list — this name is
one word away from #6 (`passes_star_gate`) but answers a *completely
different question*: not "does this move pass," but "is the whole
tech-discipline mechanism even switched on this ply." Two functions this
close in name and this different in behavior is a standing trap.

- **What:** Whether tech-purchase discipline applies at all this ply.
- **How:** ARM and SAVE always gate; GROW gates only during the expansion
  window (an Expand order is live and the tribe hasn't reached
  `COMMIT_CITY_TARGET` cities yet); UNLOCK never gates.

*Why this name:* drops "star gate" (see #6) and states directly what the
boolean answers — is the discipline mechanism active — with no overlap
against #6's name.

---

## B. Same layer, same problem, not explicitly named

Found while reading through the files above for Section A. Lower priority
than Section A (nobody named these), included because the reasoning is the
same.

### 9. `read_map` → `census_explored_terrain`
**File:** `search/archetype.rs:156` (private)

- **What:** A fog-of-war-honest terrain and metal census over everything
  the player has explored — open-field fraction, rough-terrain fraction,
  metal-tile count.
- **How:** One pass over explored tiles, tallying terrain-type counts and
  a resource lookup; no search, no ranking — a straight tally.

*Why this name:* "read_map" could mean almost any operation touching the
map; "census" says what kind of read this specifically is.

### 10. `covers` → `unit_covers_threat`
**File:** `combat.rs:549`

- **What:** Whether a specific unit satisfies a specific threatened city's
  coverage requirement (in strike range or holding a load-bearing tile).
- **How:** Checks the unit's distance and movement against the threat's
  reach radius, distinguishing "can respond in time" from "already
  standing on the contested tile."

*Why this name:* `covers` as a bare predicate name reads fine locally but
is a landmine at any call site far from its definition — `unit.covers()`
could mean almost anything. Spelling out the object (`threat`) removes the
ambiguity.

---

## C. A vocabulary inconsistency, not a single rename

`select_playstyle`'s name uses **"playstyle."** The type it returns and
sets is `Archetype`. The doc comments throughout `archetype.rs` and
`oracle_macro.rs` call the same thing a **"lane."** Three names for one
concept, used interchangeably depending on which comment or function you're
reading:

- `select_playstyle()` returns `Option<Archetype>`
- `tribe_lane_prior()`, `lane_techs()`, `SaveLane` all say "lane"
- The type declaration itself says `Archetype`

This isn't fixable by renaming one function — it needs a single word picked
and applied consistently across the type name, the function names, and the
comments. Flagging it here rather than picking one myself, since committing
to `Archetype` vs. `Lane` vs. `Playstyle` project-wide is a bigger call than
any single function rename above. My own preference, for what it's worth,
would be **"lane"** — it's already the dominant word in the doc comments,
and it reads more naturally in a sentence ("committed to the ForgeGiants
lane") than "archetype" does.

---

## D. Housekeeping: the HTML report is stale in four places

Not a naming issue — these four ledger entries in the "Three-Tier Playing
Layout" artifact (16 Aug 2026) describe functions that no longer exist
under those names, from work done earlier in today's session:

| Report says | Current reality |
|---|---|
| `city_threats` (`defense.rs:501`) — a second, independent threat model | Absorbed into `city_risks` (EXP_ELO_054, Aug 16). `defense.rs` is now `combat.rs`. |
| `prediction::predict_villages` — a second, independent village guesser | Merged into `guess_villages` (`belief/prediction.rs`) earlier this session. |
| `guessed_village_sites` (`oracle_macro.rs:904`) | Same merge — folded into `guess_villages` alongside `predict_villages`. |
| `predict_enemy_capitals` / `get_border_clouds` | Retired (Phase 2 of the reorg) in favor of `BeliefState`'s real posterior. |

Every `oracle_macro.rs:N` line number in the report is also stale — the
file dropped from ~3688 lines to under 700 today, with most of its content
split into `search/archetype.rs`, `search/goal_aux.rs`, `economy.rs`,
`movement.rs`, and `combat.rs`. The report will need a fuller pass (new
locations, the four merges above, whatever renames get adopted from this
proposal) rather than a line-number patch — worth doing as one pass once
the renames above are decided, not before.

---

## What actually shipped

All of Section A and B, plus Section C's "lane" consolidation (Verdi's
call), plus a `SaveLane` → `SaveTarget` rename not in the original proposal
— needed to free up "lane" as the one canonical name once `Archetype`
became `Lane`, since the two types would otherwise have sat in the same
files under confusingly similar names. 207/207 lib tests + full
integration suite green after each commit. Section D's HTML report was
regenerated in place (same artifact URL) with every rename, location, and
the merged-model sections (city_risks/city_threats, the two village
guessers) reflected, and its "Computes" column restructured to the
requested What:/How: format throughout chapter 8.
