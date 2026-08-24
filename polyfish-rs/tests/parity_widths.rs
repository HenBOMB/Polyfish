//! Cross-language width parity between the Rust and Python network definitions.
//!
//! `src/ai/network.rs` and `train.py` read and write the same
//! `model.safetensors`, so every shape either side builds must agree. This
//! reads the Python source and diffs it against the Rust constants; it asserts
//! AGREEMENT, never a literal, so a coordinated architecture change stays
//! green. The `aux_*` heads are deliberately Python-only and are not compared.

use polyfish::ai::features::{MAP_SIZE, NUM_CHANNELS, RawFeatures};
use polyfish::ai::mapper::NUM_MOVE_OPTIONS;
use polyfish::ai::network::NUM_ACTION_TYPES;
use regex::Regex;
use std::collections::HashMap;

const TRAIN_PY: &str = "train.py";
const NETWORK_RS: &str = "src/ai/network.rs";

fn read(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Every capture of `pattern` in `src`, which must be non-empty and unanimous.
fn unanimous(src: &str, rel: &str, what: &str, pattern: &str) -> String {
    let re = Regex::new(pattern).unwrap();
    let found: Vec<String> = re
        .captures_iter(src)
        .map(|c| c[1].to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    match found.len() {
        0 => panic!("{rel}: no match for {what} (pattern {pattern}) — the parser is stale"),
        1 => found.into_iter().next().unwrap(),
        _ => panic!("{rel}: {what} disagrees with itself: {found:?}"),
    }
}

fn unanimous_usize(src: &str, rel: &str, what: &str, pattern: &str) -> usize {
    let raw = unanimous(src, rel, what, pattern);
    raw.parse()
        .unwrap_or_else(|_| panic!("{rel}: {what} is not a number: {raw}"))
}

/// Resolve a Python layer width that may be a literal or a module constant.
fn resolve(consts: &HashMap<&str, usize>, what: &str, token: &str) -> usize {
    if let Ok(n) = token.parse::<usize>() {
        return n;
    }
    *consts.get(token).unwrap_or_else(|| {
        panic!("{TRAIN_PY}: {what} is `{token}`, which is not a module-level integer constant")
    })
}

#[test]
fn rust_and_python_network_widths_agree() {
    let py = read(TRAIN_PY);
    let rs = read(NETWORK_RS);

    let py_action_const = unanimous_usize(
        &py,
        TRAIN_PY,
        "NUM_ACTION_TYPES",
        r"(?m)^NUM_ACTION_TYPES\s*=\s*(\d+)\s*$",
    );
    let consts: HashMap<&str, usize> = [("NUM_ACTION_TYPES", py_action_const)].into();

    let py_action_head = resolve(
        &consts,
        "pi_action width",
        &unanimous(
            &py,
            TRAIN_PY,
            "pi_action width",
            r"self\.pi_action\s*=\s*nn\.Linear\([^,]+,\s*([A-Za-z_0-9]+)\s*\)",
        ),
    );
    let py_option_head = resolve(
        &consts,
        "pi_option width",
        &unanimous(
            &py,
            TRAIN_PY,
            "pi_option width",
            r"self\.pi_option\s*=\s*nn\.Linear\([^,]+,\s*([A-Za-z_0-9]+)\s*\)",
        ),
    );
    let py_channels = unanimous_usize(
        &py,
        TRAIN_PY,
        "SPATIAL_CHANNELS",
        r"(?m)^\s*SPATIAL_CHANNELS\s*=\s*(\d+)",
    );
    let py_player_dim = unanimous_usize(
        &py,
        TRAIN_PY,
        "PLAYER_STATE_DIM",
        r"(?m)^\s*PLAYER_STATE_DIM\s*=\s*(\d+)",
    );
    let py_map = unanimous_usize(&py, TRAIN_PY, "MAP_SIZE", r"(?m)^\s*MAP_SIZE\s*=\s*(\d+)");
    let py_filters = unanimous_usize(&py, TRAIN_PY, "filters", r"self\.filters\s*=\s*(\d+)");
    let py_res_blocks = unanimous_usize(
        &py,
        TRAIN_PY,
        "ResBlock count",
        r"self\.res_blocks\s*=\s*nn\.ModuleList\(\s*\[\s*ResBlock\([^)]*\)\s+for\s+_\s+in\s+range\(\s*(\d+)\s*\)",
    );
    let py_gn_groups = unanimous_usize(
        &py,
        TRAIN_PY,
        "GroupNorm groups",
        r"nn\.GroupNorm\(\s*(\d+)\s*,",
    );

    let rs_filters = unanimous_usize(
        &rs,
        NETWORK_RS,
        "FILTERS",
        r"(?m)^const FILTERS\s*:\s*usize\s*=\s*(\d+)\s*;",
    );
    let rs_res_blocks = unanimous_usize(
        &rs,
        NETWORK_RS,
        "RES_BLOCKS",
        r"(?m)^const RES_BLOCKS\s*:\s*usize\s*=\s*(\d+)\s*;",
    );
    let rs_gn_groups = unanimous_usize(
        &rs,
        NETWORK_RS,
        "GN_GROUPS",
        r"(?m)^const GN_GROUPS\s*:\s*usize\s*=\s*(\d+)\s*;",
    );

    let mismatches: Vec<String> = [
        ("action_type head", NUM_ACTION_TYPES, py_action_head),
        ("move_option head", NUM_MOVE_OPTIONS, py_option_head),
        ("spatial channels", NUM_CHANNELS, py_channels),
        (
            "player state dim",
            RawFeatures::PLAYER_STATE_DIM,
            py_player_dim,
        ),
        ("map size", MAP_SIZE, py_map),
        ("trunk filters", rs_filters, py_filters),
        ("ResBlock count", rs_res_blocks, py_res_blocks),
        ("GroupNorm groups", rs_gn_groups, py_gn_groups),
    ]
    .iter()
    .filter(|(_, rust, python)| rust != python)
    .map(|(what, rust, python)| format!("  {what}: network.rs {rust} vs train.py {python}"))
    .collect();

    assert!(
        mismatches.is_empty(),
        "Rust and Python network definitions disagree; they share model.safetensors:\n{}",
        mismatches.join("\n")
    );

    assert_eq!(
        rs_filters % rs_gn_groups,
        0,
        "GroupNorm({rs_gn_groups}) does not divide {rs_filters} filters"
    );
}

/// The parsed Python constants must be the ones actually fed to `PolyZeroNet`,
/// and the map must stay square — `features.rs` has one `MAP_SIZE` for both axes.
#[test]
fn python_model_is_built_from_the_parsed_constants() {
    let py = read(TRAIN_PY);
    let re = Regex::new(
        r"PolyZeroNet\(\s*SPATIAL_CHANNELS\s*,\s*PLAYER_STATE_DIM\s*,\s*MAP_SIZE\s*,\s*MAP_SIZE\s*\)",
    )
    .unwrap();
    assert!(
        re.is_match(&py),
        "{TRAIN_PY}: PolyZeroNet is no longer constructed as \
         PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE); \
         the constants this test parses may no longer be the ones in use"
    );
}
