//! EXP_ELO_028 Phase 1c goal-priced in-tree shaping constants (Aug 2026
//! taxonomy split out of reward.rs to keep every file under ~1000 lines): the
//! painted macro goal's actuator prices, plus `save_progress`, the v10 SAVE
//! ramp helper sized against them. Pure data + one small pure function — no
//! logic changes from the original reward.rs, just relocated. Re-exported
//! through `reward` so existing `crate::ai::reward::X` call sites keep
//! resolving.

use crate::states::GameState;

// ---- EXP_ELO_028 Phase 1c: goal-priced in-tree shaping ------------------
// The painted macro goal gets an actuator: each stance/order prices the
// resource conversion it names, as a potential on the goal-holder's side
// only (the opponent's goal is unknown at search time). Sized like
// SHAPE_PURSUIT_PER_TILE — large enough to flip a decisive Q gap (~0.15-0.2
// normalized through score_norm≈600-700).

/// Score-equivalents per star-per-turn of income while stance is GROW.
pub const SHAPE_GOAL_SPT: f32 = 150.0;
/// Extra score-equivalents per star of living army while stance is ARM
/// (on top of the game score's 5·cost).
pub const SHAPE_GOAL_ARM_PER_COST: f32 = 50.0;
/// v9: score-equivalents per star-per-turn of income while stance is ARM.
/// Half `SHAPE_GOAL_SPT` on purpose — army stays the dominant term under ARM,
/// but income stops being invisible for the 85% of post-t10 plies ARM holds.
/// FIRST FIT: dial against the measured dq median before trusting it (every
/// first fit in this file has overshot ~2x).
pub const SHAPE_GOAL_ARM_SPT: f32 = 75.0;
/// Score-equivalents per tile of closed distance toward a painted EXPAND
/// target. Summed over targets; an achieved (self-owned) target holds the
/// full cap so the final capture banks its step instead of cliffing -CAP.
pub const SHAPE_GOAL_EXPAND_PER_TILE: f32 = 200.0;
/// Score-equivalents per OWNED environment-recommended tech (GoalAux) —
/// buying the map-fit tech banks this in-tree; off-fit tech banks nothing.
pub const SHAPE_GOAL_TECH_FIT: f32 = 150.0;
/// EXP_ELO_052: score-equivalents per road tile still needed to link a city
/// to the capital. A connection pays +1 pop to the city and +1 to the
/// capital, but only the LAST tile of the path collects it — so every tile
/// before it lost its ballot to a harvest and no city ever connected (0.00
/// at t10 over 96 games). Paying down the remaining count gives each tile on
/// the path the share of that reward it actually earns.
pub const SHAPE_GOAL_CONNECT: f32 = 120.0;
/// v2.4 scout term: score-equivalents per explored tile while GROW holds,
/// no EXPAND target is known, and expansion is unfinished — pays the
/// "find your village" step the audit showed nothing was paying for.
pub const SHAPE_GOAL_SCOUT: f32 = 25.0;
/// v2.4: extra proximity-tiles banked when an EXPAND target is achieved
/// (on top of the held cap) — makes the final Capture-vs-Step choice a
/// ~0.4-normalized landslide instead of one more step of gradient.
pub const SHAPE_GOAL_EXPAND_DONE: f32 = 2.0;
/// Score-equivalents per living Rider while the rider push is on (open
/// terrain + active EXPAND).
pub const SHAPE_GOAL_RIDER: f32 = 100.0;
/// v3→v6: score-equivalents per STAR OF COST of living lane/overlay-
/// preferred units (GoalAux.preferred_units). Was 100 flat per head, which
/// made a Knight (8★) worth 17.5 Φ/star against a Defender's 38 from the
/// SAME overlay — the measured reason the knight lane researched but never
/// converted. 33/cost keeps the cost-3 units (Rider/Archer/Defender)
/// numerically unchanged (99 ≈ 100) while heavies price per star invested
/// (Knight 264, Catapult 264, Giant 330). Stacks with SHAPE_GOAL_RIDER.
pub const SHAPE_GOAL_LANE_PER_COST: f32 = 33.0;
/// v4: per-quadrant cap on scout-term tiles — tiles in a fresh quadrant keep
/// paying after a covered quadrant has gone flat (audit: half of games left
/// a quadrant unvisited).
pub const SCOUT_QUADRANT_CAP: i32 = 20;
/// v4: one-time score-equivalents per map corner explored (lighthouse →
/// monument progress; audit: 29/64 games touched zero corners).
pub const SHAPE_GOAL_LIGHTHOUSE: f32 = 120.0;
/// v4 (amended per Verdi — encourage, never hard-gate): score-equivalents per
/// Explorer reward taken, scaled by hidden_frac² (see the term). Re-sized
/// Jul 31 from the measured root Q gap (--dump-reward-choices, 166 modal
/// Explorer/Workshop plies): Workshop led by median +0.26 normalized at
/// hidden≥0.5 and the old 150·h term (≈+0.12 effective at the horizon) lost
/// 86% of those plies. Iterated against the measured dq shift (~0.085 per
/// 100): 1000 overshot (83% take, SPT@t20 −5, wr −14pp matched); 600 sat at
/// the exact flip point (dq −0.004, 52% take). 700 lands the dark-map take
/// in the registered ≥60% band at roughly half of 1000's economic cost.
pub const SHAPE_GOAL_EXPLORER: f32 = 700.0;
/// v4: extra Explorer lift when an UNREVEALED map corner sits within the
/// walk's plausible reach of the city. The 12-step fog-biased walk is
/// deterministic — an exact corner hit is already priced through
/// SHAPE_GOAL_LIGHTHOUSE in the simulated child — so this credits the
/// near-miss chance the simulation can't see. Scaled with the main term
/// (Jul 31 re-size, ~same ratio as the old 150:60).
pub const SHAPE_GOAL_EXPLORER_LIGHTHOUSE: f32 = 230.0;
/// Chebyshev radius for "corner within explorer reach". Verdi: a centrally
/// located explorer reliably reaches one, sometimes two lighthouses — so 5
/// (center-to-corner on 11x11) is in range, and the lift scales per
/// reachable dark corner, capped at 2.
pub const EXPLORER_WALK_RANGE: i32 = 5;
/// Max dark corners credited per explorer (one, sometimes two).
pub const EXPLORER_CORNER_CAP: usize = 2;
/// v5: score-equivalents per unit of reward_pop per adjacency partner BEYOND
/// the first, for owned yield structures (Windmill/Sawmill/Forge — anything
/// in structures.rs with reward_pop > 0 and adjacent_types). Steers the tile
/// choice toward multi-partner spots (audit: 52% of windmills sat next to a
/// single farm). The first partner is the structure paying for itself — no
/// bonus; a partner-less build stays unpriced rather than penalized.
pub const SHAPE_GOAL_YIELD_ADJ: f32 = 100.0;
/// v10: fraction of `SHAPE_GOAL_YIELD_ADJ` paid for UNFILLED partner capacity.
/// Realized partners are the yield; capacity is the option on it. The PLACEMENT
/// decision needs capacity, because partners are built after the hub (measured:
/// chosen ceiling 2.00 == final level 2.00, so sites do fill in) and that is far
/// past a 6.5-ply horizon — a Phi counting only what stands today cannot tell a
/// ceiling-1 tile from a ceiling-3 one. Audit: 31 sub-optimal Forge sites, all
/// in-city and legal at the time, 12 of them with no resource trade-off at all.
pub const SHAPE_GOAL_YIELD_CAPACITY_W: f32 = 0.5;
/// v6: star-yield analog for Market (reward_stars > 0 + adjacent_types) —
/// HALF the pop analog, deliberately: each partner's +1 SPT is already paid
/// at SHAPE_GOAL_SPT through get_tribe_spt; this only sharpens the 2-3-hub
/// placement choice.
pub const SHAPE_GOAL_YIELD_ADJ_STARS: f32 = 50.0;
/// v5: option value of a standing forest in own territory (future lumber
/// hut / sawmill feed / grow line). Clearing or burning drops it, making
/// the ~1/game follow-through-free clear (audit: 10% of clears fed no
/// build, level-up, or sawmill) net-negative in-tree, while justified
/// clears still win on their follow-up's much larger payoff.
pub const SHAPE_GOAL_FOREST_STANDING: f32 = 50.0;


/// v8: multiplier on the Explorer term while the tribe still holds only its
/// capital. Measured Aug 1: the capital's first city reward was Explorer in
/// **12/12 seats** — at t0 the map is maximally dark, so `hidden²` ≈ 1 and the
/// term pays ~700-930 against Workshop's 150. That is not a distribution, it is
/// a constant. Workshop's +1 SPT is worth most at t0 (longest compounding
/// horizon), so the explorer edge is discounted for exactly the first reward
/// and left untouched afterwards — the dark-map dial Verdi measured in July
/// (1000 → wr −14pp, 600 → the flip point) is deliberately NOT reopened.
pub const SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE: f32 = 0.15;

/// v6: score-equivalents of penalty PER POINT of structurally stranded
/// progress — pop sunk into a city whose next level can no longer be
/// completed from its remaining territory resources/pop structures at ANY
/// star budget. First fit (150/city, stars-dependent) fired both
/// falsifiers: star-dependence taxed all spending (wr −12pp) while
/// depth-blindness left extra pop into stranded cities free. Depth ×
/// resource-structural has no spend-tax and prices exactly the
/// harvest-into-a-dead-end Verdi flagged. Threatened cities exempt.
pub const SHAPE_GOAL_STRANDED: f32 = 75.0;

/// v7: savings ramp — Phi paid for holding `stars/cost` of the SAVE batch.
/// Deliberately below `SHAPE_GOAL_SPT`: reaching the batch and spending it must
/// beat sitting on a full bank, or the ramp becomes a reason to hoard forever.
/// First fit — dial against the measured carried-balance and purchase rates
/// per the q-gap method (first fits have overshot ~2x every time).
pub const SHAPE_GOAL_SAVE: f32 = 100.0;

/// Fraction of the SAVE batch already secured, counting BOUGHT components and
/// not just the star balance.
///
/// v10: the ramp used to read `stars/cost` alone, so buying the lane tech —
/// the whole point of banking — dropped Φ by the price of the tech (measured
/// ~76 score-equivalents on the Forge lane at a 21-star target). It paid to
/// accumulate and charged to make progress, and inside a 6.5-ply horizon only
/// the charge is visible. Progress is monotone under plan-advancing purchases.
pub fn save_progress(
    state: &GameState,
    player: i32,
    lane: &crate::ai::oracle_macro::SaveTarget,
) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let mut banked = tribe.stars.max(0);
    if lane.tech_cost > 0
        && crate::settings::technology::has_technology(&tribe.tech_vanilla, lane.tech)
    {
        banked += lane.tech_cost;
    }
    if lane.structure_unit_cost > 0 {
        let built: i32 = tribe
            .cities
            .iter()
            .filter(|c| {
                c._territory.iter().any(|&t| {
                    crate::functions::get_structure_type_at(state, t) == Some(lane.structure)
                })
            })
            .count() as i32;
        banked += built * lane.structure_unit_cost;
    }
    (banked as f32 / lane.cost as f32).clamp(0.0, 1.0)
}

/// v10: score-equivalents for OWNING a super unit.
///
/// The level-5 reward is a binary pick: Park (+250 score AND +1 production) vs
/// a super unit (cost 10 -> +50 score). Raw score prefers Park 5:1 and only
/// ARM's `SHAPE_GOAL_ARM_PER_COST` ever overturned it. Measured park share of
/// that pick: 8-12% under ARM, but 78-83% under GROW and 88% under SAVE.
///
/// Sized against the measured totals so the pick lands where a strong human
/// puts it: ARM ~3:1 for the giant, GROW ~1.4:1, and SAVE only inverting to
/// Park once the banked plan is nearly done. Super units are reward-only
/// (`moves/reward.rs:143`), never summoned, so this cannot distort purchasing —
/// it prices the pick and the cost of losing one.
pub const SHAPE_GOAL_SUPER: f32 = 500.0;

/// EXP_ELO_040: per assigned covering unit on a Defend-ordered city — full
/// pay inside 1-turn strike reach, half in the 2-turn ring. Sized above the
/// max single-ply Expand approach gain (2 tiles × EXPAND_PER_TILE) so an
/// at-risk leash holds, while 0.5-urgency cover still loses to a 2-step
/// expansion move: intensity calibrated to the actual need.
/// EXP_ELO_050: weight on the score-equivalent expected city loss. The
/// underlying quantity is already in Φ's units (city worth), so this is a
/// trust dial, not a unit conversion — calibrate by the q-gap method against
/// the measured dq, never by guess.
/// EXP_ELO_051: measured, not guessed. At 1.0 a ply that vacated a
/// threatened city priced at −0.01 while a Research earned +0.167 — the term
/// was live but an order of magnitude too quiet to compete. Dialled against
/// the observed edge-reward distribution over 24 games.
pub const SHAPE_GOAL_CITY_RISK: f32 = 4.0;
pub const SHAPE_GOAL_DEFEND_COVER: f32 = 600.0;

/// Tile-holding pay, only while the garrison is load-bearing
/// (`DefendPlan::hold_needed`) — an unconditional term would re-create
/// garrison pinning, which the coverage design exists to avoid.
pub const SHAPE_GOAL_DEFEND_HOLD: f32 = 400.0;

/// EXP_ELO_042: symmetric offense pricing per unit pressing an
/// Attack-ordered city (sat 1.0 in strike reach, 0.5 in the 2-turn ring).
/// 500 beats the max single-ply Expand approach gain (2 × EXPAND_PER_TILE)
/// so a committed attacker never abandons a push for a village pull, and
/// sits below at-risk DEFEND_COVER+HOLD so home always outbids in a tie.
pub const SHAPE_GOAL_ATTACK_PRESS: f32 = 500.0;

/// A unit standing ON an enemy city pays PRESS × this, BY STATE-FACT —
/// independent of Attack orders, so the pay survives order flicker when a
/// co-attacker dies. Makes stepping off a siege a ≥250 Φ loss.
pub const SHAPE_GOAL_SIEGE_HOLD_MULT: f32 = 1.5;

/// How much a nearly-complete SAVE plan damps `SHAPE_GOAL_SUPER`. At full
/// progress the giant bonus drops to 40%, flipping the pick back to Park —
/// which is right when +1 production finishes a plan this turn.
///
/// Deliberately keyed to `save_progress` ONLY, not `completion_progress`: under
/// GROW the city nearest a level-up drives completion toward 1.0 exactly at the
/// reward ply, so using it would damp the bonus at every single level-5 pick
/// and quietly rebuild the pathology this term exists to fix.
pub const SHAPE_GOAL_SUPER_ECON_DAMP: f32 = 0.6;

/// v7: completion BONUS — pays progress toward a REACHABLE level (see
/// `completion_progress`). Held below `SHAPE_GOAL_SPT` so a level-up, which
/// zeroes the bonus and banks the SPT jump, is always the better move.
pub const SHAPE_GOAL_COMPLETION: f32 = 75.0;

/// v6: approach weight for an enemy-taken village painted as a retake
/// target — pays slightly under a free village (the defender-adjusted
/// odds); the recapture itself banks the full DONE bonus.
pub const SHAPE_GOAL_RETAKE_W: f32 = 0.75;
/// v6: a contested EXPAND target (visible enemy standing on it) pays the
/// nearest second unit this fraction of the approach gradient — exactly
/// one converger, so a squatter can be killed and the tile still taken.
pub const SHAPE_GOAL_CONTEST_SECOND: f32 = 0.5;
/// v6: score-equivalents per living unit up to a state-derived cap
/// (min(cities+1, BODY_CAP_MAX)) while GROW holds and there is still map
/// to take — the early bodies that scout and grab villages. Flat per head:
/// warriors ARE the desired bodies; the cap kills summon spam. 75 clears
/// the measured −0.118 summon Q deficit without doubling it (dial lesson).
pub const SHAPE_GOAL_BODY: f32 = 75.0;
/// Cap on bodies the GROW body term pays for.
pub const BODY_CAP_MAX: usize = 3;

