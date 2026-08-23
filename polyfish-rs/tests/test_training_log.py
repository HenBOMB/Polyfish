#!/usr/bin/env python3
"""Tests for training_log.py — the canonical experiment record.

Every number in `training_log.csv` and both dashboard stores passes through
here, and the failures this covers are silent: a duplicated CSV row and an
erased history both look like ordinary output. Stdlib `unittest` on purpose,
matching tests/test_ladder.py.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _InTempDir(unittest.TestCase):
    """training_log.py addresses every store by a bare relative path."""

    def setUp(self):
        import training_log

        self.tl = training_log
        self._prev = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev)
        self._tmp.cleanup()

    def quietly(self, fn, *args, **kwargs):
        """Run fn with stderr captured; returns (result, stderr)."""
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = fn(*args, **kwargs)
        return result, buf.getvalue()


class MetricsSidecarTest(_InTempDir):
    """#37: a sidecar left in place is re-read by the next iteration, so a
    producer that exits without writing one duplicates the previous
    iteration's numbers into the CSV under a new iteration."""

    def test_train_sidecar_is_consumed_by_the_read(self):
        with open(self.tl.TRAIN_METRICS_PATH, "w") as f:
            json.dump({"loss": 1.25}, f)

        self.assertEqual(self.tl.parse_train_output()["loss"], 1.25)
        self.assertFalse(os.path.exists(self.tl.TRAIN_METRICS_PATH))

    def test_second_iteration_cannot_reread_the_first_sidecar(self):
        with open(self.tl.TRAIN_METRICS_PATH, "w") as f:
            json.dump({"loss": 1.25}, f)
        self.tl.parse_train_output()

        # train.py wrote nothing this time (its no-data path exits 0).
        self.assertEqual(self.tl.parse_train_output(""), {})

    def test_self_play_sidecar_is_consumed_by_the_read(self):
        with open(self.tl.SELF_PLAY_METRICS_PATH, "w") as f:
            json.dump({"avg_score": 42, "games_file": "games_1.safetensors"}, f)

        self.assertEqual(self.tl.parse_self_play_output()["avg_score"], 42)
        self.assertFalse(os.path.exists(self.tl.SELF_PLAY_METRICS_PATH))

    def test_stdout_fallback_still_parses_when_no_sidecar_exists(self):
        text = 'METRICS: {"loss": 0.5}'
        self.assertEqual(self.tl.parse_train_output(text)["loss"], 0.5)


class DashboardStoreTest(_InTempDir):
    """#37: both stores were read-modify-written in place and reset to {} on a
    decode error, so a crash mid-dump erased every run's history — and neither
    file is reconstructible from the CSV."""

    def test_corrupt_store_is_kept_aside_not_discarded(self):
        with open(self.tl.MOVES_PATH, "w") as f:
            f.write('{"1": {"1": {"tr')  # truncated mid-dump

        _, err = self.quietly(self.tl.update_moves_by_turn, "2", 1, {"0": 3})

        self.assertIn("kept as", err)
        with open(self.tl.MOVES_PATH + ".corrupt") as f:
            self.assertEqual(f.read(), '{"1": {"1": {"tr')
        with open(self.tl.MOVES_PATH) as f:
            self.assertEqual(json.load(f), {"2": {"1": {"0": 3}}})

    def test_existing_runs_survive_a_new_write(self):
        with open(self.tl.MOVES_PATH, "w") as f:
            json.dump({"run_a": {"1": {"0": 1}}}, f)

        self.tl.update_moves_by_turn("run_b", 7, {"0": 9})

        with open(self.tl.MOVES_PATH) as f:
            store = json.load(f)
        self.assertEqual(store["run_a"], {"1": {"0": 1}})
        self.assertEqual(store["run_b"], {"7": {"0": 9}})

    def test_write_leaves_no_temp_file(self):
        self.tl.update_moves_by_turn("run_a", 1, {"0": 1})
        self.assertFalse(os.path.exists(self.tl.MOVES_PATH + ".tmp"))

    def test_save_store_replaces_atomically(self):
        # os.replace never exposes a partial file: the reader either sees the
        # whole old store or the whole new one.
        self.tl._save_store(self.tl.MOVES_PATH, {"a": 1})
        self.tl._save_store(self.tl.MOVES_PATH, {"b": 2})
        with open(self.tl.MOVES_PATH) as f:
            self.assertEqual(json.load(f), {"b": 2})

    def test_missing_store_reads_as_empty(self):
        self.assertEqual(self.tl._load_store(self.tl.MOVES_PATH), {})

    def test_non_dict_store_reads_as_empty(self):
        with open(self.tl.MOVES_PATH, "w") as f:
            json.dump([1, 2, 3], f)
        self.assertEqual(self.tl._load_store(self.tl.MOVES_PATH), {})


if __name__ == "__main__":
    unittest.main()
