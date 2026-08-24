#!/usr/bin/env python3
"""The checkpoint-key contract for the two inference backends CI cannot run.

`metal_network.rs` and `tch_network.rs` look every weight up by name at
graph-build time and *panic* on a miss (metal_network.rs:307-310,
tch_network.rs:81-84). Neither can be compiled on Linux, so audit E1 — metal
asking for BatchNorm-era `bn1.weight` while checkpoints store `gn1.weight` —
shipped and sat undetected. `examples/{tch,metal}_parity.rs` do assert the key
set, but both need Apple hardware, so nothing automated covers it.

This does the name half statically: it reads the keys each backend asks for
straight out of the Rust source and checks them against the state_dict
`init_model.py` writes. No cargo, no Apple hardware, no libtorch.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import json
import os
import re
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import torch  # noqa: F401

    import train

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a torch-less runner
    HAVE_TORCH = False
    train = None

BACKENDS = ("src/ai/metal_network.rs", "src/ai/tch_network.rs")
# Helpers that take a weight *prefix* and load `<prefix>.weight` + `<prefix>.bias`.
PAIR_HELPERS = ("linear", "conv2d", "group_norm")
# Helpers that take a full tensor name.
BARE_HELPERS = ("const_natural", "const_shape", "get")
CALL = re.compile(r"self\.(" + "|".join(PAIR_HELPERS + BARE_HELPERS) + r")\s*\(")
LIT = re.compile(r'"([^"]+)"')


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _arglist(src, open_paren):
    """Text between a call's parens, respecting nesting and string literals."""
    depth, i, n = 0, open_paren, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            i += 1
            while i < n and src[i] != '"':
                i += 2 if src[i] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return src[open_paren + 1 : i]
        i += 1
    raise AssertionError("unbalanced call expression")


def required_keys(rel):
    """Every safetensors key this backend loads by name."""
    src = _read(rel)
    n_blocks = int(re.search(r"NUM_RES_BLOCKS: usize = (\d+)", src).group(1))
    # `let p = format!("res_blocks.{i}")` and friends.
    binds = dict(re.findall(r'let\s+(\w+)\s*=\s*format!\("([^"]+)"\)', src))
    keys = set()
    for m in CALL.finditer(src):
        helper = m.group(1)
        args = _arglist(src, m.end() - 1)
        lit = LIT.search(args)
        if lit is None:
            continue  # argument is a variable: a generic helper forwarding a prefix
        name = lit.group(1)
        for var, expansion in binds.items():
            name = name.replace("{" + var + "}", expansion)
        if "{prefix}" in name:
            continue  # inside a helper body; the caller supplies the prefix
        expanded = (
            [name.replace("{i}", str(i)) for i in range(n_blocks)]
            if "{i}" in name
            else [name]
        )
        for key in expanded:
            assert "{" not in key, f"unresolved format placeholder {key!r} in {rel}"
            if helper in PAIR_HELPERS:
                keys.update((key + ".weight", key + ".bias"))
            else:
                keys.add(key)
    assert keys, f"extracted no weight keys from {rel} — the parser has drifted"
    return keys


def parity_table_keys(rel):
    """The key set `examples/{tch,metal}_parity.rs::expected_shapes()` commits to."""
    src = _read(rel)
    body = src[src.index("fn expected_shapes()") :]
    body = body[: body.index("\nfn ", 1)]
    keys = set(re.findall(r'"([A-Za-z_][A-Za-z_0-9.]*)"\s*\.into\(\)', body))
    leaves = re.findall(r'\(\s*"([a-z0-9_.]+)"\s*,\s*vec!', body)
    n_blocks = int(re.search(r"RES_BLOCKS: usize = (\d+)", src).group(1))
    for i in range(n_blocks):
        keys.update(f"res_blocks.{i}.{leaf}" for leaf in leaves)
    return keys


def init_model_state_dict_keys():
    """Key set of the model `init_model.py` writes, built the way it builds it."""
    src = _read("init_model.py")
    dims = {
        name: int(re.search(rf"^\s*{name} = (\d+)", src, re.M).group(1))
        for name in ("MAP_SIZE", "SPATIAL_CHANNELS", "PLAYER_STATE_DIM")
    }
    model = train.PolyZeroNet(
        dims["SPATIAL_CHANNELS"], dims["PLAYER_STATE_DIM"], dims["MAP_SIZE"], dims["MAP_SIZE"]
    )
    return set(model.state_dict().keys())


def safetensors_header_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k for k in header if k != "__metadata__"}


class BackendKeyContractTest(unittest.TestCase):
    def test_metal_and_tch_ask_for_the_same_keys(self):
        """metal_network.rs is documented as a byte-for-byte op mirror of
        tch_network.rs (metal_network.rs:12-14), so the two key sets must be
        identical. This is the assertion that fires on an E1-style half-rename."""
        metal, tch = (required_keys(b) for b in BACKENDS)
        self.assertEqual(metal, tch)

    def test_no_backend_looks_up_a_batchnorm_key(self):
        """Audit E1: `bn1.weight` / `bn2.weight` are pre-GroupNorm names no
        checkpoint has stored since migrate_model.py:25-30 started rejecting them.
        The BatchNorm-era *rejection guards* (metal_network.rs:267,
        tch_network.rs:52) are string comparisons, not lookups, so they are not
        matched here."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                offenders = sorted(
                    k for k in required_keys(backend) if re.search(r"(^|\.)bn\d", k)
                )
                self.assertEqual(offenders, [])

    def test_parity_examples_cover_exactly_what_the_backends_load(self):
        """The Mac-only parity gates carry a hand-written table. If it drifts from
        the code, running them on Apple hardware proves nothing."""
        required = required_keys(BACKENDS[0])
        for example in ("examples/tch_parity.rs", "examples/metal_parity.rs"):
            with self.subTest(example=example):
                self.assertEqual(parity_table_keys(example), required)

    @unittest.skipUnless(HAVE_TORCH, "building the reference state_dict requires torch")
    def test_every_required_key_exists_in_a_fresh_model(self):
        """The contract that actually matters: a model straight out of
        init_model.py must carry every tensor these backends load. Extra keys are
        the training-only aux heads (v_progress / v_ownership), ignored by design."""
        available = init_model_state_dict_keys()
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(sorted(required_keys(backend) - available), [])

    def test_every_required_key_exists_in_the_local_checkpoint(self):
        """Same contract against the real model.safetensors when one is present.
        It is gitignored, so this skips in CI and covers a dev box instead."""
        path = os.path.join(ROOT, "model.safetensors")
        if not os.path.exists(path):
            self.skipTest("model.safetensors not present")
        available = safetensors_header_keys(path)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(sorted(required_keys(backend) - available), [])


if __name__ == "__main__":
    unittest.main()
