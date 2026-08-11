# Mapgen research: our generator vs the real Polytopia generator

*(Aug 10, 2026. Trigger: playing Xin-xi in the simulator feels metal-starved compared to
the real game.)*

**STATUS: FIXED same day — see ledger entry MAPGEN_001 in
`hypothesis_driven_improvements.md` for the change list, post-fix measurements, and the
catalog of measurement discontinuities it causes. Divergence items 1, 2, 4 and 6 below
are resolved, 3 partially (Xin-xi's guarantee added; the other tribes' guarantees are
still undocumented upstream); 5 (village spacing) and 7 (per-tribe water) deliberately
left.**

## TL;DR — one interpretation bug, not many wrong constants

`mapgen.rs` was built from the Polytopia wiki's Map Generation table, and its numbers are
all *there*: fruit 18%/6%, crop 18%/6%, animal 19%/6%, metal 11%/3%, mountain 14%,
forest 38%, field 48%. But the wiki's resource percentages are **fractions of all land
tiles** (joint probabilities: 14% of land is mountain, 11% of land is
mountain-with-metal), while our port rolls them as **per-matching-terrain-tile
conditionals** (each mountain gets metal with p=0.11). The correct conditionals are the
joint divided by the terrain share:

| resource | real P(resource \| eligible tile), inner | ours | deficit |
|---|---|---|---|
| metal on mountain | 11/14 ≈ **0.79** (last exact code constant: **0.85**, patch 2.0.20) | 0.11 | **~7×** |
| game on forest | 19/38 = **0.50** | 0.19 | ~2.6× |
| fruit on field | 18/48 = **0.375** | 0.18 | ~2.1× |
| crop on field | 18/48 = **0.375** | 0.18 | ~2.1× |
| fish on shallow water | 0.50 | 0.50 | ✓ correct (it's the one rate the wiki lists conditionally) |

Outer ring (distance 2 from a village) = inner × **1/3** ("border expansion" factor,
constant across all game eras); our 0.03/0.06 outer values inherit the same misread.
The terrain shares themselves (14/38/48) *are* conditionals-on-land, and our code uses
them correctly — which is why our terrain looks right and only resources feel starved.

The deficit is worst for metal because mountains are the rarest eligible terrain, so the
joint↔conditional gap is largest. For Xin-xi specifically the real game plays
`0.79–0.85 × 1.5 ≥ 1` → **every mountain in inner city territory effectively carries
metal**; ours gives 0.165. That is precisely the "I expected way more metal" feeling.

## Empirical confirmation (measured identically on both sides)

Ground truth: three turn-0 real-Steam-game captures made by polyfish-mod, in
`polyfish-rs/replays/`:

| capture | version | map | tribes |
|---|---|---|---|
| `anjiian-atoll_1774877964` | 111 | Drylands 16×16 | Oumaji, Cymanti, Vengir, **XinXi**, Kickoo, Hoodrick |
| `adventure-of-assha_1774996907` | 114 | Lakes 16×16 | Yadakk, Elyrion |
| `basin-games_1774838316` | 115 | Lakes 14×14 | Cymanti, AiMo |

Ours: 300 generated Drylands 11×11 maps per tribe pair (temporary probe binary, since
deleted). Measurement: share of eligible terrain bearing the resource, bucketed by
Chebyshev distance to the nearest village/capital site. Real-capture cells are small-n
(±15pp); the direction is consistent across all three captures and both map types.

| resource | real d1 (per capture) | real d2 | ours d1 | ours d2 |
|---|---|---|---|---|
| metal | 75%, 57%, 75% (pooled 19/27 = **70%**) | 45%, 43%, 20% (pooled **38%**) | **14%** (no metal tribe) / 21–22% (Xin-xi in game) | **4–5%** |
| game | 46%, 45%, 59% | 12%, 7%, 6% | 25–29% | 5–8% |
| fruit | 31%¹, 56%, 60%¹ | 19%, 19%, 21% | 41% (Imperius ×2) / 21–24% (no fruit tribe) | 7–11% |
| crop | 28%, 36%, 8%² | 8%, 8%, 0% | 12–24% (mix-dependent) | 4–7% |
| fish | 100% (n=2), 43%, 54% | –, 21%, 20% | 100% (Kickoo ponds) | – |

¹ no fruit-bonus tribe on the map — the real base rate is high for everyone.
² Cymanti+AiMo (AiMo carries a 0.1 crop penalty in the real game too).

Matched-mix spot check — Cymanti+AiMo: real 75% metal d1 / 20% d2 vs ours, same pair,
15% / 4%. Neither tribe has a metal multiplier, so the real high rate is the *base*.
Both sides show **0 metal beyond distance 2** (real 0/12, ours 0/thousands): the
"resources only within 2 tiles of a village" rule is correctly ported — it's the rates
inside that radius that are wrong. Xin-xi close-up: our Xin-xi capital averages **0.65
metal within radius 2**; the real Xin-xi capture had **5** (every near-capital mountain
had ore). Measured rates sit above the raw constants on both sides because village
radii overlap and our r2 pass re-rolls failed r1 tiles; since the measurement is
identical on both sides, the comparison stands.

## How the real algorithm works (provenance per era)

The community record splits into three eras; the constants above are era-consistent.

**Era 1 (pre-Moonrise, 2019–2020) — QuasiStellar's generator**
([github.com/QuasiStellar/Polytopia-Map-Generator](https://github.com/QuasiStellar/Polytopia-Map-Generator)).
Its faithfulness is proven by Midjiwan's own 2.0.20 patch notes listing five *before*
values (mountain .15, metal .5, Xin-xi mountain 1.5, Xin-xi metal 1, Quetzali metal .1)
that exactly match the repo, published nine months earlier. Algorithm: random land →
3× cellular-automaton smoothing → capitals maximize min-distance → tribe climate
flood-fill → biome rolls (forest 0.4, mountain 0.15 × tribe mults) → villages (flat
land, ≥3 apart, never on edge; capitals stamp radius-1 = "initial territory", radius-2 =
"border expansion") → resources via `proc()`: full rate at distance ≤1, ×1/3 at
distance 2, **zero elsewhere**; metal 0.5, game 0.5, fish 0.5, whale 0.4, fruit/crop
0.5 with a mutual-exclusion factor `(1−other/2)` → 0.375 effective each. Xin-xi had
**no** capital-resource guarantee yet (Imperius/Bardur/Elyrion/Kickoo/Zebasi did).

**Era 2 (Moonrise 2.0.20, Oct 2020) — the metal buff.** Verbatim patch notes
([Steam](https://store.steampowered.com/news/app/874390/view/2865936253729801250)):
mountain .15→.2, **metal .5→.85**, Xin-xi mountain 1.5→2, Xin-xi metal 1→1.5, Quetzali
metal .1→1. Later: 2.0.58 (Sep 2021) moved terrain to a quota system (deterministic
proportions instead of independent rolls).

**Era 3 (current: Path of the Ocean 2.8+ / Aquarion rework, 2023–25).** Sources: the
wiki's [Map Generation](https://polytopia.fandom.com/wiki/Map_Generation) page and
Espark's decompilation-derived
[per-tribe rates table](https://static.wikia.nocookie.net/supertribes/images/5/58/Resource_rates_by_tribe.png/revision/latest)
(2025). This is the era our port targeted, and it matches almost everything else in
`mapgen.rs`: phase order (capitals → villages → terrain → resources → ruins/starfish),
quadrant capital placement on Drylands/Lakes/Archipelago/WaterWorld, suburbs, the
pre-terrain village coefficients (0.3/0.1), island-village counts per size, ruins
counts (Tiny 4 … Massive 23), starfish ≈ 1 per 25 water tiles, and the entire tribe
multiplier table (Xin-xi 1.5 mountain/1.5 metal, Vengir 2.0 metal, Bardur 0.8 forest/0
crop, Cymanti 1.2 mountain, Oumaji 0.2 forest/0.2 animal, etc. — ours agrees with
Espark's image on every entry I checked). Xin-xi starting stars 7 ✓ (2025 Balance
Pass). The wiki also states verbatim: "All resources spawn only within two tiles of
cities or villages" and "The standard spawn rate for fish is 50% among shallow water
tiles."

Unresolved upstream (flagged by the research, doesn't change conclusions): whether
Xin-xi mountain is 1.5 or 2.0 today (two independent 2024–25 sources say 1.5); whether
the modern 48/38/14 terrain shares are code constants or quota-system outputs; exact
current metal constant (0.80 vs 0.85 — both round to the wiki's 11%/14%).

## Divergence list for mapgen.rs (beyond the headline bug)

1. **`get_resource_prob()` semantics** — the joint-vs-conditional misread above. Correct
   inner base conditionals: metal ~0.8, game 0.5, fruit 0.375, crop 0.375, fish 0.5;
   outer = ×1/3.
2. **Fish outer rate** — ours is flat 0.50 at distance 2; real is 0.5/3 ≈ 0.17 (our one
   rate that's too *high*; measured real d2 fish ≈ 20% confirms).
3. **Capital guarantee table** — modern game guarantees every tribe a starting resource;
   **Xin-xi's is metal** (Espark's "Starting Resource" row). Ours guarantees nothing for
   Xin-xi (also nothing for Vengir, Hoodrick, Oumaji, Quetzali, Yadakk, AiMo, Luxidoor,
   Polaris).
4. **Primary-resource cap of 3** in capital r1 (`max_spawns`) is a repo invention (the
   comment cites a user report). With correct conditionals the real game does produce
   4–6 primary resources at some capitals; the cap exists to patch a symptom that came
   from the ×2 Imperius multiplier sitting on a mis-scaled base.
5. **Village spacing** — ours enforces ≥3 apart; the modern wiki says ≥2. (Era 1 was ≥3;
   likely changed with the modern rewrite. Low impact, affects village density.)
6. **Our r2 resource pass re-rolls r1 tiles** (radius-2 square includes the radius-1
   tiles; a failed inner roll gets a second outer-rate roll from the same village).
   Real-game structure is one classification per tile (inner XOR outer). Small upward
   bias today, but NOT negligible once the base rates are corrected: 0.8 inner with a
   0.27 re-roll compounds to ~0.85+ before village overlap. Any fix must decide whether
   to keep or remove the double roll, not just swap constants.
7. **No per-tribe water mechanism** — the real game gives Kickoo/Aquarion extra water
   in their climate zone (Era 1: 0.4/0.3 tile replacement; Era 3 lists Kickoo 2.0
   water, partially bugged per the wiki). Our water comes only from the map-type land
   ratio. Immaterial on Drylands training maps, real on Lakes/Continents.

## Who consumes this generator

- `self_play.rs` — every training game (Drylands 11×11) the net has ever seen was
  generated with ~7×-starved metal, ~2.6×-starved game, ~2×-starved fruit/crop.
- `main.rs` — the simulator the user plays (startup fallback, `reset_game`, replay
  reconstruction from `initial_seed`).

## Addendum (Aug 10, later): placement / Forge-spot follow-up

Question: beyond rates, does the real game *place* mountains differently (ridges/clumps),
making Forge spots (flat tile, ≥2 adjacent metal mountains) easier to find?

**No clustering mechanism exists in the real game.** All three real captures test
consistent with uniform scatter — mean mountain-neighbors-per-mountain vs a
uniform-shuffle null over eligible land: z = −1.07 (anjiian), −0.84 (assha), +0.00
(basin). Era-1 code rolled terrain i.i.d. per tile; the modern quota system pins
*counts*, and the captures show placement stays effectively uniform. (2 of 3 real maps
lean slightly *anti*-clustered — not significant at n=3; unresolved observation only.)

**Post-fix, our Forge-spot supply is at least comparable to the real game, possibly
overshooting** (300 maps/config, same metric as the captures):

| config | spots ≥2 adj metal per 100 land | maps with zero | best-spot ≥3 |
|---|---|---|---|
| real anjiian (Drylands 16, 6 tribes) | 2.8 (7 spots, n=1 map) | — | no (best=2) |
| ours, same mix/size post-fix | 5.4 (13.7/map) | 0% | 78% of maps |
| ours XinXi+Kickoo Tiny | 7.6 (9.1/map) | 0% | 76% |
| ours Imperius+Bardur Tiny | 3.8 (4.6/map) | 11% | 33% |
| real assha (Lakes 16, mt-poor tribes) | 0.5 | — | no |
| real basin (Lakes 14, mt-rich tribes) | 6.1 | — | yes (best=3) |

Xin-xi capitals: a ≥2-metal Forge spot within Chebyshev 2 of the capital in **100% of
games**. Slight over-clustering appears only in Xin-xi configs (obs 1.17 vs null 1.00;
Imperius+Bardur 0.88 vs 0.86) — that's the capital metal carve firing when the ring has
<2 natural mountains, whose real-game placement mechanics are undocumented. Honest open
item, direction = we may overshoot spot quality near capitals.

**Residual real placement differences (measured, small):** no quota system → per-map
mountain-count variance ±4-5 (real pins counts; 11% of our Imperius+Bardur maps still
have zero good spots, 0% of Xin-xi maps); village spacing ours ≥3 vs real ≥2 (real has
*more* villages yet measures *fewer* spots, so this cannot explain a perceived deficit).

**The perceived deficit itself was the rates bug**: the simulator server running during
this session started pre-fix (0.65 metal near a Xin-xi capital → adjacent metal *pairs*
essentially never occurred). Restart the server to see post-fix maps.

## Implications (not acted on)

- Mining/Smithery/Forge economics are systematically underrepresented in the training
  distribution; the eco_plan hub lanes and any metal-dependent strategy are learned
  against a resource landscape ~7× poorer than the real game's.
- Fixing is a small change (divide-by-terrain-share on five numbers, the guarantee-table
  row, and a decision on the double-roll in item 6), but it shifts the training
  distribution: pre-register per the
  hypothesis ledger before any run consumes it, and expect replay/seed continuity to
  break for economy-dependent gauges (seed-770425 harness measurements are not
  comparable across the change).
- The mid-game real captures (`saved_state.json`, turn 11) show harvested/mined tiles
  keep their metal rate signature — turn-0-only analysis was still used throughout to
  be safe.
