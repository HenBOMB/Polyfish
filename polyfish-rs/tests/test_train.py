#!/usr/bin/env python3
"""Tests for train.py — the trainer the training loop actually runs.

Audit T3 recorded that train.py, the primary trainer, had no test
infrastructure at all. These cover the helpers whose failure mode is *silent*:
a holdout that leaks reports a good number for an overfitting model, and a
spatial pad that lands channels in the wrong place feeds the net garbage
without ever raising.

Requires torch (train.py imports it at module scope); skipped where it is
absent, so a bare CI runner still executes the rest of the suite.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import os
import re
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


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class HoldoutSplitTest(unittest.TestCase):
    """The holdout is the only evidence that can separate underfitting from
    overfitting (audit M5), so its two invariants are load-bearing: it must not
    leak, and membership must not move between iterations."""

    FILES = [f"games_{i}.safetensors" for i in range(400)]

    def test_split_is_a_partition(self):
        kept, held = train.split_holdout(self.FILES, 0.15)
        self.assertEqual(sorted(kept + held), sorted(self.FILES))
        self.assertEqual(set(kept) & set(held), set())

    def test_holdout_is_roughly_the_requested_fraction(self):
        _, held = train.split_holdout(self.FILES, 0.15)
        self.assertGreater(len(held) / len(self.FILES), 0.08)
        self.assertLess(len(held) / len(self.FILES), 0.25)

    def test_membership_is_stable_across_calls(self):
        first = train.split_holdout(self.FILES, 0.15)[1]
        second = train.split_holdout(list(reversed(self.FILES)), 0.15)[1]
        self.assertEqual(sorted(first), sorted(second))

    def test_membership_ignores_the_directory(self):
        # Files migrate root -> archive/ as the buffer rolls. If the path
        # decided membership, a held-out game would re-enter training on the
        # move and inflate the next reading.
        for f in self.FILES[:40]:
            self.assertEqual(
                train.is_holdout_file(f, 0.15),
                train.is_holdout_file(os.path.join("archive", f), 0.15),
            )

    def test_zero_fraction_disables_the_holdout(self):
        kept, held = train.split_holdout(self.FILES, 0.0)
        self.assertEqual(held, [])
        self.assertEqual(kept, self.FILES)

    def test_never_returns_an_empty_training_set(self):
        # A tiny buffer can hash entirely into the holdout; training on nothing
        # is worse than not holding out.
        kept, held = train.split_holdout(self.FILES[:1], 1.0)
        self.assertEqual(kept, self.FILES[:1])
        self.assertEqual(held, [])


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class TeacherBufferTest(unittest.TestCase):
    """#36: teachers are mixed into every iteration and never rotate out, while
    holdout membership is a stable function of the basename. Splitting the
    combined list therefore withheld a random ~15% of the teacher set from
    fitting for the whole campaign, and put static known-good positions into
    `value_r2_holdout` — the series that is supposed to say how the net
    generalizes on fresh self-play."""

    FRAC = 0.15
    SELF_PLAY = [f"games_{i}.safetensors" for i in range(200)]
    TEACHERS = [f"teachers/games_pro_{i}.safetensors" for i in range(200)]

    def setUp(self):
        # Without this the rest of the class would pass on a fixture that never
        # reaches the case at all.
        self.assertTrue(
            [f for f in self.TEACHERS if train.is_holdout_file(f, self.FRAC)],
            "fixture contains no teacher that hashes into the holdout",
        )

    def test_a_teacher_that_hashes_into_the_holdout_still_trains(self):
        kept, held = train.partition_buffer(self.SELF_PLAY, self.TEACHERS, self.FRAC)
        for f in self.TEACHERS:
            self.assertIn(f, kept)
            self.assertNotIn(f, held)

    def test_splitting_the_combined_list_is_what_lost_them(self):
        """The shape of the defect, pinned so the old call cannot come back."""
        _, held = train.split_holdout(self.SELF_PLAY + self.TEACHERS, self.FRAC)
        self.assertTrue(set(held) & set(self.TEACHERS))
        self.assertFalse(
            set(train.partition_buffer(self.SELF_PLAY, self.TEACHERS, self.FRAC)[1])
            & set(self.TEACHERS)
        )

    def test_the_holdout_is_self_play_only(self):
        _, held = train.partition_buffer(self.SELF_PLAY, self.TEACHERS, self.FRAC)
        self.assertTrue(held)
        self.assertTrue(set(held) <= set(self.SELF_PLAY))

    def test_the_self_play_split_is_the_one_it_would_get_alone(self):
        _, held = train.partition_buffer(self.SELF_PLAY, self.TEACHERS, self.FRAC)
        self.assertEqual(sorted(held), sorted(train.split_holdout(self.SELF_PLAY, self.FRAC)[1]))

    def test_every_file_lands_on_exactly_one_side(self):
        kept, held = train.partition_buffer(self.SELF_PLAY, self.TEACHERS, self.FRAC)
        self.assertEqual(sorted(kept + held), sorted(self.SELF_PLAY + self.TEACHERS))
        self.assertEqual(set(kept) & set(held), set())

    def test_teachers_cannot_stand_in_for_a_withheld_self_play_buffer(self):
        # The guard against an empty training set now sees the self-play buffer
        # alone. Combined, a teacher that hashed out kept it non-empty, so on a
        # small buffer the guard never fired and the iteration fit teachers only
        # while withholding every self-play file it had.
        held_sp = [f for f in self.SELF_PLAY if train.is_holdout_file(f, self.FRAC)][:1]
        kept_teacher = [f for f in self.TEACHERS if not train.is_holdout_file(f, self.FRAC)][:1]
        self.assertTrue(held_sp and kept_teacher)
        kept, held = train.partition_buffer(held_sp, kept_teacher, self.FRAC)
        self.assertEqual(held, [])
        self.assertEqual(sorted(kept), sorted(held_sp + kept_teacher))

    def test_a_teacher_only_buffer_still_trains(self):
        self.assertEqual(
            train.partition_buffer([], self.TEACHERS, self.FRAC), (self.TEACHERS, [])
        )

    def test_no_teachers_is_the_plain_split(self):
        self.assertEqual(
            train.partition_buffer(self.SELF_PLAY, [], self.FRAC),
            train.split_holdout(self.SELF_PLAY, self.FRAC),
        )


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class PadSpatialTest(unittest.TestCase):
    """Legacy 136-channel data is zero-padded to 142. Channels were appended at
    the end of the layout, so padding must go at the end and nowhere else."""

    C, S = 142, 11

    def test_pads_4d_at_the_end(self):
        x = torch.ones(3, 136, self.S, self.S)
        out = train.pad_spatial(x, self.C, self.S)
        self.assertEqual(tuple(out.shape), (3, self.C, self.S, self.S))
        self.assertTrue(torch.all(out[:, :136] == 1))
        self.assertTrue(torch.all(out[:, 136:] == 0))

    def test_pads_flat_at_the_end(self):
        area = self.S * self.S
        x = torch.ones(3, 136 * area)
        out = train.pad_spatial(x, self.C, self.S)
        self.assertEqual(tuple(out.shape), (3, self.C * area))
        self.assertTrue(torch.all(out[:, : 136 * area] == 1))
        self.assertTrue(torch.all(out[:, 136 * area :] == 0))

    def test_flat_and_4d_agree(self):
        area = self.S * self.S
        flat = torch.arange(2 * 136 * area, dtype=torch.float32).reshape(2, 136 * area)
        as4d = flat.reshape(2, 136, self.S, self.S)
        self.assertTrue(
            torch.equal(
                train.pad_spatial(flat, self.C, self.S).reshape(2, self.C, self.S, self.S),
                train.pad_spatial(as4d, self.C, self.S),
            )
        )

    def test_current_width_is_untouched(self):
        x = torch.randn(2, self.C, self.S, self.S)
        self.assertTrue(torch.equal(train.pad_spatial(x, self.C, self.S), x))

    def test_preserves_dtype(self):
        x = torch.ones(1, 136, self.S, self.S, dtype=torch.float16)
        self.assertEqual(train.pad_spatial(x, self.C, self.S).dtype, torch.float16)


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class D4Test(unittest.TestCase):
    """AUGMENT_D4 is off by default (a mid-run enable was measured to collapse
    play), but the transform itself must still be a group action or an
    enable would silently corrupt data rather than merely destabilise it."""

    def test_four_rotations_are_the_identity(self):
        x = torch.randn(2, 3, 11, 11)
        y = x
        for _ in range(4):
            y = train.apply_d4(y, 1, False)
        self.assertTrue(torch.equal(x, y))

    def test_flip_is_an_involution(self):
        x = torch.randn(2, 3, 11, 11)
        self.assertTrue(torch.equal(train.apply_d4(train.apply_d4(x, 0, True), 0, True), x))

    def test_identity_element(self):
        x = torch.randn(2, 3, 11, 11)
        self.assertTrue(torch.equal(train.apply_d4(x, 0, False), x))

    def test_acts_only_on_the_trailing_dims(self):
        x = torch.randn(2, 3, 11, 11)
        self.assertEqual(tuple(train.apply_d4(x, 1, True).shape), tuple(x.shape))


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class ScheduleTest(unittest.TestCase):
    def test_cosine_spans_the_whole_run(self):
        base, total = 0.002, 500
        self.assertAlmostEqual(train.cosine_lr(base, 0, total), base, places=9)
        self.assertAlmostEqual(train.cosine_lr(base, total, total), 1e-5, places=9)
        mid = train.cosine_lr(base, total // 2, total)
        self.assertLess(mid, base)
        self.assertGreater(mid, 1e-5)

    def test_is_monotonically_decreasing(self):
        vals = [train.cosine_lr(0.002, s, 100) for s in range(0, 101, 5)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_clamps_outside_the_run(self):
        self.assertAlmostEqual(train.cosine_lr(0.002, -5, 100), train.cosine_lr(0.002, 0, 100))
        self.assertAlmostEqual(train.cosine_lr(0.002, 500, 100), train.cosine_lr(0.002, 100, 100))


@unittest.skipUnless(HAVE_TORCH, "train.py requires torch")
class BatchReportTest(unittest.TestCase):
    def test_reports_every_batch_when_there_are_few(self):
        self.assertEqual(train.batch_report_indices(4), {1, 2, 3, 4})

    def test_caps_and_stays_in_range(self):
        idx = train.batch_report_indices(1000, max_reports=10)
        self.assertLessEqual(len(idx), 10)
        self.assertIn(1, idx)
        self.assertIn(1000, idx)
        self.assertTrue(all(1 <= i <= 1000 for i in idx))

    def test_no_batches_reports_nothing(self):
        self.assertEqual(train.batch_report_indices(0), set())


class WidthParityTest(unittest.TestCase):
    """The Python half of the Rust<->Python width contract.

    tests/parity_widths.rs asserts this from the Rust side; this asserts it from
    the side that actually writes model.safetensors, and it runs without a Rust
    toolchain. Both sides must agree or the file does not round-trip.
    """

    @staticmethod
    def _rust_const(path, name):
        with open(os.path.join(ROOT, path)) as f:
            src = f.read()
        m = re.search(rf"(?:pub )?const {name}: usize = (\d+)", src)
        if m:
            return int(m.group(1))
        m = re.search(rf"{name}: usize = (\d+)", src)
        assert m, f"{name} not found in {path}"
        return int(m.group(1))

    @staticmethod
    def _py_const(name):
        # SPATIAL_CHANNELS / PLAYER_STATE_DIM are function-local in train(), so
        # allow leading indentation rather than anchoring at column 0.
        with open(os.path.join(ROOT, "train.py")) as f:
            src = f.read()
        m = re.search(rf"^[ \t]*{name} = (\d+)", src, re.M)
        assert m, f"{name} not found in train.py"
        return int(m.group(1))

    @staticmethod
    def _rust_tripwire(name):
        """Read a width off network.rs's const-assert block. NUM_CHANNELS is
        computed in features.rs (= CH_MEM_END), so the literal the Rust side
        commits to lives in that tripwire, not in the declaration."""
        with open(os.path.join(ROOT, "src/ai/network.rs")) as f:
            src = f.read()
        m = re.search(rf"assert!\((?:\w+::)?{name} == (\d+)\)", src)
        assert m, f"tripwire for {name} not found in network.rs"
        return int(m.group(1))

    def test_action_head_width_agrees(self):
        self.assertEqual(
            self._rust_const("src/ai/network.rs", "NUM_ACTION_TYPES"),
            self._py_const("NUM_ACTION_TYPES"),
        )

    def test_spatial_channels_agree(self):
        self.assertEqual(self._rust_tripwire("NUM_CHANNELS"), self._py_const("SPATIAL_CHANNELS"))

    def test_move_option_width_agrees(self):
        rust = self._rust_const("src/ai/mapper.rs", "NUM_MOVE_OPTIONS")
        with open(os.path.join(ROOT, "train.py")) as f:
            src = f.read()
        m = re.search(r"self\.pi_option = nn\.Linear\(self\.filters, (\d+)\)", src)
        self.assertIsNotNone(m, "pi_option width not found in train.py")
        self.assertEqual(rust, int(m.group(1)))

    def test_player_state_dim_agrees(self):
        with open(os.path.join(ROOT, "src/ai/features.rs")) as f:
            src = f.read()
        m = re.search(r"PLAYER_STATE_DIM: usize = (\d+)", src)
        self.assertIsNotNone(m, "PLAYER_STATE_DIM not found in features.rs")
        declared = int(m.group(1))
        self.assertEqual(declared, self._py_const("PLAYER_STATE_DIM"))
        self.assertEqual(declared, self._rust_tripwire("PLAYER_STATE_DIM"))


class EnvContractTest(unittest.TestCase):
    """Every env var train.py reads must be exported by run_training_loop.sh or
    be a declared optional knob (#30: TRAIN_RUN_ID/TRAIN_TOTAL_ITERS were read
    but never exported, so run scoping was dead code and the cosine LR pinned
    at its floor across campaigns). Source-level, so it runs without torch."""

    # Knobs a user sets by hand for an off-default run; the loop deliberately
    # does not export them. Adding a var here is a claim that train.py's
    # default is the production behavior.
    OPTIONAL = {
        "TRAIN_EPOCHS",
        "TRAIN_LR",
        "TRAIN_HOLDOUT_FRAC",
        "TRAIN_CHUNK_FILES",
        "TRAIN_OPTIMIZER_STATE",
        "VALUE_LOSS_WEIGHT",
        "OWNERSHIP_LOSS_WEIGHT",
        "AUGMENT_D4",
    }

    @staticmethod
    def _read(path):
        with open(os.path.join(ROOT, path)) as f:
            return f.read()

    def test_every_env_read_is_exported_or_declared_optional(self):
        reads = set(
            re.findall(r'os\.environ(?:\.get\(|\[)\s*"([A-Z0-9_]+)"', self._read("train.py"))
        )
        self.assertTrue(reads, "no os.environ reads found in train.py — regex rotted?")
        exports = set(
            re.findall(r'^\s*export ([A-Z0-9_]+)=', self._read("run_training_loop.sh"), re.M)
        )
        unaccounted = reads - exports - self.OPTIONAL
        self.assertEqual(
            unaccounted, set(),
            f"train.py reads {sorted(unaccounted)} but run_training_loop.sh never exports "
            "them and they are not on the optional-knob allowlist. Export from the loop or "
            "add to EnvContractTest.OPTIONAL — silently falling back to the default is how "
            "run scoping became dead code (#30).",
        )

    def test_run_scoping_is_exported(self):
        exports = set(
            re.findall(r'^\s*export ([A-Z0-9_]+)=', self._read("run_training_loop.sh"), re.M)
        )
        self.assertLessEqual({"TRAIN_RUN_ID", "TRAIN_TOTAL_ITERS"}, exports)


if __name__ == "__main__":
    unittest.main()
