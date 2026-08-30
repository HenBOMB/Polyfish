//! Fixed-seed matchup selection: which map seed and which tribe pair game
//! `i` of a run gets. Shared by `self_play`, `arena` and `actor_ceiling`,
//! which previously carried three near-identical copies of this (68 of 71
//! lines byte-identical between the first two, and the tribe parsers had
//! already drifted apart).
//!
//! The seed-file format is `eval_seeds.json`: `{"seeds": [{"seed": N,
//! "tribe1": "...", "tribe2": "..."}, ...]}`, tribes optional but all-or-
//! nothing per entry.

use crate::types::TribeType;

/// The 12 tribes v1 net training uses.
///
/// The four "special" tribes — Aquarion, Elyrion, Polaris, Cymanti — carry
/// bespoke unit and terrain rulesets, and are deliberately left out so
/// evaluation stays on the same distribution self-play trains on. This is a
/// training-scope decision, not an oversight: widening it means retraining,
/// not just editing a list. `self_play`'s random pool is exactly this set;
/// `arena` additionally refuses them by name (see `parse_core_tribe`).
pub const CORE_TRIBES: [TribeType; 12] = [
    TribeType::Imperius,
    TribeType::Bardur,
    TribeType::Oumaji,
    TribeType::Kickoo,
    TribeType::XinXi,
    TribeType::Zebasi,
    TribeType::AiMo,
    TribeType::Vengir,
    TribeType::Luxidoor,
    TribeType::Quetzali,
    TribeType::Hoodrick,
    TribeType::Yadakk,
];

/// Tribe name (case-insensitive) to `TribeType`, over all 16 tribes.
///
/// Used by `self_play`'s `--tribe1`/`--tribe2` and its `--seed-file` tribe
/// pins: an explicit override may name a special tribe even though the random
/// pool is `CORE_TRIBES`. Unknown names fall back to `default` with a warning
/// rather than hard-erroring.
pub fn parse_tribe(s: &str, default: TribeType) -> TribeType {
    match s.to_lowercase().as_str() {
        "imperius" => TribeType::Imperius,
        "bardur" => TribeType::Bardur,
        "oumaji" => TribeType::Oumaji,
        "kickoo" => TribeType::Kickoo,
        "xinxi" => TribeType::XinXi,
        "zebasi" => TribeType::Zebasi,
        "aimo" => TribeType::AiMo,
        "vengir" => TribeType::Vengir,
        "luxidoor" => TribeType::Luxidoor,
        "quetzali" => TribeType::Quetzali,
        "hoodrick" => TribeType::Hoodrick,
        "yadakk" => TribeType::Yadakk,
        "aquarion" => TribeType::Aquarion,
        "elyrion" => TribeType::Elyrion,
        "polaris" => TribeType::Polaris,
        "cymanti" => TribeType::Cymanti,
        _ => {
            eprintln!("Unknown tribe {}, using {:?}", s, default);
            default
        }
    }
}

/// `CORE_TRIBES` only — `arena`'s parser. A special-tribe name is recognized
/// but refused, so the fallback is reported as the deliberate v1 exclusion it
/// is rather than as an unknown name. See `CORE_TRIBES`.
pub fn parse_core_tribe(s: &str, default: TribeType) -> TribeType {
    let t = parse_tribe(s, default);
    if CORE_TRIBES.contains(&t) {
        return t;
    }
    eprintln!("Tribe {t:?} is out of scope for v1 net training (see CORE_TRIBES), using {default:?}");
    default
}

/// Picks a (t1, t2) pair for one game. If `--tribe1`/`--tribe2` are given they
/// pin that slot for every game; otherwise a distinct pair is sampled from
/// `all_tribes` using `rng`, so each caller with a different rng gets a
/// different pair.
pub fn pick_tribes(
    rng: &mut impl rand::Rng,
    all_tribes: &[TribeType],
    tribe1_arg: &Option<String>,
    tribe2_arg: &Option<String>,
) -> (TribeType, TribeType) {
    use rand::seq::SliceRandom;
    let t1 = match tribe1_arg {
        Some(s) => parse_tribe(s, TribeType::Imperius),
        None => *all_tribes.choose(rng).unwrap(),
    };
    let t2 = match tribe2_arg {
        Some(s) => parse_tribe(s, TribeType::Oumaji),
        None => loop {
            let t = *all_tribes.choose(rng).unwrap();
            if t != t1 {
                break t;
            }
        },
    };
    (t1, t2)
}

/// Resolves one game's tribe pair. Precedence, highest wins:
/// 1. CLI `--tribe1`/`--tribe2` — if either is set, defers entirely to
///    `pick_tribes` (which honors the CLI pin(s) and randomly fills any
///    slot left unset), exactly as before `--seed-file` tribes existed.
/// 2. The `--seed-file` entry's own tribe1/tribe2 pair (`seed_file_tribes`),
///    when neither CLI flag is set — pins both slots for this game
///    without touching `rng`.
/// 3. `pick_tribes`' random draw off this game's own seed, when neither of
///    the above applies.
pub fn resolve_tribes(
    rng: &mut impl rand::Rng,
    all_tribes: &[TribeType],
    tribe1_arg: &Option<String>,
    tribe2_arg: &Option<String>,
    seed_file_tribes: Option<(TribeType, TribeType)>,
) -> (TribeType, TribeType) {
    if tribe1_arg.is_some() || tribe2_arg.is_some() {
        return pick_tribes(rng, all_tribes, tribe1_arg, tribe2_arg);
    }
    if let Some(pair) = seed_file_tribes {
        return pair;
    }
    pick_tribes(rng, all_tribes, tribe1_arg, tribe2_arg)
}

#[derive(serde::Deserialize)]
struct RawSeedEntry {
    seed: i64,
    #[serde(default)]
    tribe1: Option<String>,
    #[serde(default)]
    tribe2: Option<String>,
}

#[derive(serde::Deserialize)]
struct SeedFile {
    seeds: Vec<RawSeedEntry>,
}

/// One loaded --seed-file entry: a map seed plus an optional pinned tribe
/// pair (see eval_seeds.json). `tribes` is `Some` only when both tribe1
/// and tribe2 are present on that entry.
#[derive(Clone, Copy)]
pub struct SeedEntry {
    pub seed: i64,
    pub tribes: Option<(TribeType, TribeType)>,
}

/// Loads a fixed seed list (see eval_seeds.json). Errors rather than
/// silently wrapping if it's shorter than the count requested.
///
/// `parse` is injected because the callers disagree on purpose: `self_play`
/// passes `parse_tribe` (all 16), `arena` passes `parse_core_tribe` (the 12
/// of `CORE_TRIBES`).
pub fn load_seed_file(
    path: &str,
    needed: usize,
    parse: impl Fn(&str, TribeType) -> TribeType,
) -> anyhow::Result<Vec<SeedEntry>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("reading --seed-file {path}: {e}"))?;
    let parsed: SeedFile = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("parsing --seed-file {path}: {e}"))?;
    anyhow::ensure!(
        parsed.seeds.len() >= needed,
        "--seed-file {path} has {} seeds but {needed} were requested",
        parsed.seeds.len()
    );
    parsed
        .seeds
        .into_iter()
        .map(|e| {
            let tribes = match (e.tribe1.as_deref(), e.tribe2.as_deref()) {
                (Some(t1), Some(t2)) => Some((
                    parse(t1, TribeType::Imperius),
                    parse(t2, TribeType::Oumaji),
                )),
                (None, None) => None,
                _ => anyhow::bail!(
                    "--seed-file {path}: seed {} has one of tribe1/tribe2 set but not the other",
                    e.seed
                ),
            };
            Ok(SeedEntry { seed: e.seed, tribes })
        })
        .collect()
}

/// Game i's map seed: `seed_list[i]` when a fixed list is given, else the
/// legacy `base_seed + i` derivation.
pub fn seed_for_game(i: usize, base_seed: u64, seed_list: Option<&[i64]>) -> i64 {
    match seed_list {
        Some(list) => list[i],
        None => (base_seed + i as u64) as i64,
    }
}

/// Game i's --seed-file-pinned tribe pair, if that entry specifies one.
/// Parallel accessor to `seed_for_game` -- same indexing, but for the
/// tribe pair instead of the map seed.
pub fn tribes_for_game(i: usize, entries: Option<&[SeedEntry]>) -> Option<(TribeType, TribeType)> {
    entries.and_then(|list| list[i].tribes)
}

#[cfg(test)]
#[path = "eval_seeds_tests.rs"]
mod tests;
