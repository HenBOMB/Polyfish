#!/usr/bin/env python3
"""Tests for ladder.py — the gauge's statistics.

Every strength verdict in this project is drawn from these functions, so an
error here is invisible in training and shows up only as a wrong conclusion in
`hypothesis_driven_improvements.md`. Stdlib `unittest` on purpose: no scientific
stack is pinned for the training env (requirements.txt), and CI runs bare
python3.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WilsonTest(unittest.TestCase):
    def setUp(self):
        import ladder

        self.ladder = ladder

    def test_interval_stays_inside_unit_range(self):
        for wr in (0.0, 0.01, 0.5, 0.99, 1.0):
            for n in (1, 8, 64, 1000):
                lo, hi = self.ladder._wilson(wr, n)
                self.assertGreaterEqual(lo, 0.0, f"wr={wr} n={n}")
                self.assertLessEqual(hi, 1.0, f"wr={wr} n={n}")
                self.assertLessEqual(lo, hi)

    def test_zero_games_is_maximally_uninformative(self):
        self.assertEqual(self.ladder._wilson(0.5, 0), [0.0, 1.0])

    def test_interval_narrows_with_more_games(self):
        widths = [self.ladder._wilson(0.33, n)[1] - self.ladder._wilson(0.33, n)[0]
                  for n in (16, 64, 256, 1024)]
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_known_value(self):
        # 21 wins of 64 at p=0.33: the audit's worked example for M3.
        lo, hi = self.ladder._wilson(0.33, 64)
        self.assertAlmostEqual(lo, 0.2273, places=3)
        self.assertAlmostEqual(hi, 0.4519, places=3)

    def test_half_width_reproduces_the_audit_figure(self):
        # M3's headline: a 64-game reading resolves to about +/-11.5pp.
        self.assertAlmostEqual(self.ladder._half_width(0.33, 64), 11.23, places=2)


class NormalQuantileTest(unittest.TestCase):
    def setUp(self):
        import ladder

        self.ladder = ladder

    def test_matches_reference_quantiles(self):
        # Beasley-Springer-Moro should be good to ~1e-9 against the true values.
        for tail, expected in ((0.025, 1.959963985), (0.05, 1.644853627),
                               (0.20, 0.841621234), (0.005, 2.575829304)):
            self.assertAlmostEqual(self.ladder._z_from_tail(tail), expected, places=6)

    def test_symmetric_about_the_median(self):
        for tail in (0.001, 0.01, 0.1, 0.3):
            self.assertAlmostEqual(
                self.ladder._z_from_tail(tail), -self.ladder._z_from_tail(1.0 - tail), places=6
            )

    def test_median_is_zero(self):
        self.assertAlmostEqual(self.ladder._z_from_tail(0.5), 0.0, places=9)


class RequiredGamesTest(unittest.TestCase):
    def setUp(self):
        import ladder

        self.ladder = ladder

    def test_registered_bar_needs_far_more_than_the_gauge_spends(self):
        # EXP_ELO_002 registered +8pp at a ~33% baseline and read it off 64
        # games. This is the number M3 says was never computed.
        n = self.ladder.required_games(0.33, 0.08)
        self.assertGreater(n, 500)
        self.assertLess(n, 650)

    def test_smaller_effects_need_more_games(self):
        ns = [self.ladder.required_games(0.33, d) for d in (0.20, 0.12, 0.08, 0.05)]
        self.assertEqual(ns, sorted(ns))

    def test_more_power_needs_more_games(self):
        lo = self.ladder.required_games(0.33, 0.08, power=0.50)
        hi = self.ladder.required_games(0.33, 0.08, power=0.95)
        self.assertLess(lo, hi)

    def test_no_effect_is_undetectable(self):
        self.assertIsNone(self.ladder.required_games(0.33, 0.0))

    def test_clamps_at_the_boundaries(self):
        # Should not raise on a baseline the search can actually produce.
        self.assertIsNotNone(self.ladder.required_games(0.0, 0.05))
        self.assertIsNotNone(self.ladder.required_games(1.0, -0.05))


class WinRateTest(unittest.TestCase):
    def setUp(self):
        import ladder

        self.ladder = ladder

    def test_draws_count_as_half(self):
        self.assertAlmostEqual(self.ladder._win_rate(10, 10, 0), 0.5)
        self.assertAlmostEqual(self.ladder._win_rate(0, 0, 10), 0.5)
        self.assertAlmostEqual(self.ladder._win_rate(5, 10, 10), 0.4)

    def test_no_games_is_zero_not_a_crash(self):
        self.assertEqual(self.ladder._win_rate(0, 0, 0), 0.0)


class VerdictTest(unittest.TestCase):
    """The freeze and plateau gates must read the interval, not the point."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LADDER_FILE"] = os.path.join(self.tmp.name, "ladder.json")
        for mod in ("ladder",):
            sys.modules.pop(mod, None)
        import ladder

        self.ladder = ladder

    def tearDown(self):
        del os.environ["LADDER_FILE"]
        sys.modules.pop("ladder", None)
        self.tmp.cleanup()

    def _record(self, wins, losses, draws=0, iteration=1):
        class Args:
            pass

        a = Args()
        a.run_id = "t"
        a.iteration = iteration
        a.wins, a.losses, a.draws = wins, losses, draws
        a.avg_score_model = a.avg_score_opponent = 0.0
        a.mcts, a.gumbel_k, a.eval_backend = 64, 16, "candle"
        a.wins_p1 = a.wins_p2 = None
        a.stats_dir = None
        a.kind = "gauge"
        a.opponent = None
        data = self.ladder._load()
        reading = self.ladder._append_reading(data, a, "gauge", data["anchors"][-1])
        return data, reading

    def test_a_lucky_small_sample_does_not_clear_the_freeze_bar(self):
        # 85% of 20 games looks like a freeze on the point estimate; its lower
        # bound is nowhere near 0.80.
        _, reading = self._record(17, 3)
        self.assertGreaterEqual(reading["win_rate"], self.ladder.FREEZE_WR)
        self.assertLess(reading["win_rate_ci"][0], self.ladder.FREEZE_WR)

    def test_a_large_decisive_sample_does_clear_it(self):
        _, reading = self._record(380, 20)
        self.assertGreaterEqual(reading["win_rate_ci"][0], self.ladder.FREEZE_WR)

    def test_every_reading_records_its_own_resolution(self):
        _, reading = self._record(21, 40, 3)
        self.assertIn("resolves_pp", reading)
        self.assertAlmostEqual(
            reading["resolves_pp"],
            self.ladder._half_width(reading["win_rate"], reading["games"]),
            places=6,
        )

    @staticmethod
    def _series(*wins):
        return [{"kind": "gauge", "opponent": "greedy", "games": 64,
                 "wins": w, "losses": 64 - w, "draws": 0} for w in wins]

    def test_plateau_needs_non_overlapping_pooled_halves(self):
        flat = self._series(*([20] * 8))
        self.assertTrue(self.ladder._plateau(flat))
        climbing = self._series(*([10] * 4 + [55] * 4))
        self.assertFalse(self.ladder._plateau(climbing))

    def test_plateau_needs_a_full_window(self):
        short = self._series(*([20] * (self.ladder.PLATEAU_WINDOW - 1)))
        self.assertFalse(self.ladder._plateau(short))

    def test_pooling_beats_a_single_reading_on_resolution(self):
        # Why the plateau test pools: 8 x 64 games resolves ~2.8x tighter than
        # any one of them, which is the only reason the gate is meaningful at
        # this budget.
        one = self.ladder._wilson(20 / 64, 64)
        pooled_wr, pooled_n = self.ladder._pool(self._series(*([20] * 8)))
        pooled = self.ladder._wilson(pooled_wr, pooled_n)
        self.assertLess(pooled[1] - pooled[0], one[1] - one[0])


class PowerCommandTest(unittest.TestCase):
    def test_cli_emits_parseable_json(self):
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(
            [sys.executable, os.path.join(root, "ladder.py"), "power",
             "--baseline", "0.33", "--games", "64"],
            capture_output=True, text=True, check=True,
        )
        d = json.loads(out.stdout)
        self.assertEqual(d["at_games"], 64)
        self.assertGreater(d["games_per_reading"], d["at_games"])
        self.assertAlmostEqual(d["resolves_pp"], 11.23, places=2)


if __name__ == "__main__":
    unittest.main()
