#!/usr/bin/env python3
"""Tests for ladder.py — the gauge's statistics.

Every strength verdict in this project is drawn from these functions, so an
error here is invisible in training and shows up only as a wrong conclusion in
`hypothesis_driven_improvements.md`. Stdlib `unittest` on purpose: no scientific
stack is pinned for the training env (requirements.txt), and CI runs bare
python3.

    python3 -m unittest discover -s tests -p 'test_*.py'
"""
import contextlib
import io
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

    def test_a_flat_series_strikes(self):
        self.assertTrue(self.ladder._plateau(self._series(*([20] * 8))))

    def test_a_declining_series_strikes(self):
        self.assertTrue(self.ladder._plateau(self._series(30, 28, 26, 24, 22, 20, 18, 16)))

    def test_a_steady_climb_does_not_strike(self):
        """The regression this gate was rewritten for. +1pp per reading is
        +8pp across the window — EXP_ELO_002's registered effect size — and the
        interval-overlap rule struck on it every time, stopping the run two
        gauge cycles into a real improvement."""
        climb = self._series(21, 22, 22, 23, 24, 24, 25, 26)
        self.assertFalse(self.ladder._plateau(climb))
        # ...and it is not that the climb is obvious: the pooled halves' Wilson
        # intervals still overlap, which is exactly what the old rule read.
        first = self.ladder._wilson(*self.ladder._pool(climb[:4]))
        second = self.ladder._wilson(*self.ladder._pool(climb[4:]))
        self.assertTrue(first[0] <= second[1] and second[0] <= first[1])

    def test_a_big_jump_does_not_strike(self):
        self.assertFalse(self.ladder._plateau(self._series(*([10] * 4 + [55] * 4))))

    def test_both_conditions_are_required(self):
        """The rule is a conjunction, so either half can veto a strike."""
        # Halves flat-or-down, but the window trends up (a late surge).
        late_surge = self._series(20, 24, 24, 22, 10, 10, 20, 46)
        self.assertLessEqual(
            self.ladder._pool(late_surge[4:])[0], self.ladder._pool(late_surge[:4])[0]
        )
        self.assertGreater(self.ladder._slope(late_surge), 0.0)
        self.assertFalse(self.ladder._plateau(late_surge))

        # Halves up, but the window trends down (an early spike carrying them).
        early_spike = self._series(50, 10, 10, 10, 20, 20, 20, 22)
        self.assertGreater(
            self.ladder._pool(early_spike[4:])[0], self.ladder._pool(early_spike[:4])[0]
        )
        self.assertLess(self.ladder._slope(early_spike), 0.0)
        self.assertFalse(self.ladder._plateau(early_spike))

    def test_slope_signs_the_trend(self):
        self.assertGreater(self.ladder._slope(self._series(10, 20, 30, 40)), 0.0)
        self.assertLess(self.ladder._slope(self._series(40, 30, 20, 10)), 0.0)
        self.assertEqual(self.ladder._slope(self._series(20, 20, 20, 20)), 0.0)
        self.assertEqual(self.ladder._slope(self._series(20)), 0.0)

    def test_plateau_needs_a_full_window(self):
        short = self._series(*([20] * (self.ladder.PLATEAU_WINDOW - 1)))
        self.assertFalse(self.ladder._plateau(short))

    def test_series_excludes_a_different_search_budget(self):
        # Ladder Elo is a function of (weights x sims). A 16-sim stint pooled
        # with 64-sim readings reads a search change as a weights change.
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 20,
             "losses": 44, "draws": 0, "budget": {"mcts": 16, "gumbel_k": 16}},
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0, "budget": {"mcts": 64, "gumbel_k": 16}},
        ]}
        series = self.ladder._gauge_series(data)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["budget"]["mcts"], 64)

    def test_series_excludes_a_different_turn_cap(self):
        # The loop varies GAUGE_MAX_TURNS with self_play's curriculum, so a
        # 10-turn-cap reading and a 45-turn-cap one are different instruments.
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 20,
             "losses": 44, "draws": 0,
             "budget": {"mcts": 64, "gumbel_k": 16, "max_turns": 10}},
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0,
             "budget": {"mcts": 64, "gumbel_k": 16, "max_turns": 45}},
        ]}
        series = self.ladder._gauge_series(data)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["budget"]["max_turns"], 45)

    def test_ramped_search_knobs_do_not_fragment_the_window(self):
        # The gauge tracks self-play's prior/sigma(Q) ramps (#32), so these
        # change every iteration by design. They are recorded, not keyed: key
        # on them and every reading is its own budget, so no plateau window
        # ever accumulates.
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 20,
             "losses": 44, "draws": 0,
             "budget": {"mcts": 64, "gumbel_k": 16, "max_turns": 45,
                        "prior_heuristic_w": 0.5, "q_weight": 0.0}},
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0,
             "budget": {"mcts": 64, "gumbel_k": 16, "max_turns": 45,
                        "prior_heuristic_w": 0.1, "q_weight": 1.0}},
        ]}
        self.assertEqual(len(self.ladder._gauge_series(data)), 2)

    def test_series_is_scoped_to_the_latest_run(self):
        # A previous campaign's readings are a different model's; pooling them
        # into this run's window judges a trend that never happened.
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "run_id": "old", "games": 64,
             "wins": 20, "losses": 44, "draws": 0},
            {"kind": "gauge", "opponent": "greedy", "run_id": "new", "games": 64,
             "wins": 30, "losses": 34, "draws": 0},
        ]}
        series = self.ladder._gauge_series(data)
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["run_id"], "new")

    def test_series_keeps_readings_with_no_run_id(self):
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "run_id": "", "games": 64,
             "wins": 20, "losses": 44, "draws": 0},
            {"kind": "gauge", "opponent": "greedy", "run_id": "", "games": 64,
             "wins": 30, "losses": 34, "draws": 0},
        ]}
        self.assertEqual(len(self.ladder._gauge_series(data)), 2)

    def test_series_keeps_legacy_readings_with_no_budget(self):
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 20,
             "losses": 44, "draws": 0},
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0},
        ]}
        self.assertEqual(len(self.ladder._gauge_series(data)), 2)

    def test_series_ignores_readings_against_another_anchor(self):
        data = {"anchors": [{"name": "greedy"}], "readings": [
            {"kind": "gauge", "opponent": "iter50", "games": 64, "wins": 20,
             "losses": 44, "draws": 0},
            {"kind": "link", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0},
            {"kind": "gauge", "opponent": "greedy", "games": 64, "wins": 30,
             "losses": 34, "draws": 0},
        ]}
        self.assertEqual(len(self.ladder._gauge_series(data)), 1)

    def test_dropped_games_are_recorded_not_hidden(self):
        class Args:
            pass

        a = Args()
        a.run_id, a.iteration = "t", 1
        a.wins, a.losses, a.draws = 20, 28, 0
        a.avg_score_model = a.avg_score_opponent = 0.0
        a.mcts, a.gumbel_k, a.eval_backend = 64, 16, "candle"
        a.wins_p1 = a.wins_p2 = None
        a.stats_dir = None
        a.games_attempted, a.games_dropped, a.unpaired_seeds = 64, 16, 8
        a.tribes = "Imperius,Bardur"
        data = self.ladder._load()
        r = self.ladder._append_reading(data, a, "gauge", data["anchors"][-1])
        self.assertEqual(r["games"], 48)
        self.assertEqual(r["games_attempted"], 64)
        self.assertEqual(r["games_dropped"], 16)
        self.assertEqual(r["unpaired_seeds"], 8)
        self.assertEqual(r["tribes"], "Imperius,Bardur")

    def _record_cmd(self, run_id, wins, losses, iteration=1, max_turns=45):
        """Drive the real `record` entry point and return its verdict JSON."""
        class Args:
            pass

        a = Args()
        a.run_id, a.iteration = run_id, iteration
        a.wins, a.losses, a.draws = wins, losses, 0
        a.avg_score_model = a.avg_score_opponent = 0.0
        a.mcts, a.gumbel_k, a.eval_backend = 64, 16, "candle"
        a.max_turns = max_turns
        a.wins_p1 = a.wins_p2 = None
        a.stats_dir = None
        a.kind, a.opponent = "gauge", None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.ladder.cmd_record(a)
        return json.loads(buf.getvalue())

    def _seed_plateaued_run(self, run_id):
        """A ladder mid-campaign: one strike on the board and a full flat
        window behind it, so the next flat reading is the stopping one."""
        data = self.ladder._load()
        data["plateau_strikes"] = 1
        data["plateau_run_id"] = run_id
        for i in range(self.ladder.PLATEAU_WINDOW):
            data["readings"].append({
                "kind": "gauge", "opponent": "greedy", "run_id": run_id,
                "iteration": i, "games": 64, "wins": 20, "losses": 44, "draws": 0,
                "budget": {"mcts": 64, "gumbel_k": 16, "max_turns": 45},
            })
        self.ladder._save(data)

    def test_a_second_strike_in_the_same_run_stops_it(self):
        self._seed_plateaued_run("runA")
        verdict = self._record_cmd("runA", 20, 44, iteration=99)
        self.assertEqual(verdict["action"], "stop")
        self.assertEqual(verdict["plateau_strikes"], self.ladder.PLATEAU_STRIKES)

    def test_a_new_run_does_not_inherit_the_previous_run_s_strike(self):
        """Defects 2 and 3 together: strikes used to persist in ladder.json and
        the window pooled across run_ids, so a fresh campaign could stop two
        readings in, on a previous model's evidence."""
        self._seed_plateaued_run("runA")
        verdict = self._record_cmd("runB", 20, 44, iteration=1)
        self.assertEqual(verdict["action"], "continue")
        self.assertEqual(verdict["plateau_strikes"], 0)

    def test_a_reading_records_the_turn_cap_it_was_played_at(self):
        self._record_cmd("runA", 20, 44, max_turns=10)
        with open(os.environ["LADDER_FILE"]) as f:
            reading = json.load(f)["readings"][-1]
        self.assertEqual(reading["budget"]["max_turns"], 10)
        self.assertEqual(
            self.ladder._budget_key(reading), (64, 16, 10)
        )

    def test_pooling_beats_a_single_reading_on_resolution(self):
        # Why the plateau test pools: 8 x 64 games resolves ~2.8x tighter than
        # any one of them, which is the only reason the gate is meaningful at
        # this budget.
        one = self.ladder._wilson(20 / 64, 64)
        pooled_wr, pooled_n = self.ladder._pool(self._series(*([20] * 8)))
        pooled = self.ladder._wilson(pooled_wr, pooled_n)
        self.assertLess(pooled[1] - pooled[0], one[1] - one[0])


class TribeScopeTest(unittest.TestCase):
    """#34: the ladder recorded self-play's shuffled training pair on a match
    arena hardcoded to an Imperius mirror, so the permanent experiment record
    carried metadata about a variable the gauge never varied."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["LADDER_FILE"] = os.path.join(self.tmp.name, "ladder.json")
        sys.modules.pop("ladder", None)
        import ladder

        self.ladder = ladder

    def tearDown(self):
        del os.environ["LADDER_FILE"]
        sys.modules.pop("ladder", None)
        self.tmp.cleanup()

    def _args(self, kind="gauge", tribes="Imperius,Imperius", iteration=1):
        class Args:
            pass

        a = Args()
        a.run_id, a.iteration = "t", iteration
        a.wins, a.losses, a.draws = 20, 44, 0
        a.avg_score_model = a.avg_score_opponent = 0.0
        a.mcts, a.gumbel_k, a.eval_backend = 64, 16, "candle"
        a.max_turns = 45
        a.wins_p1 = a.wins_p2 = None
        a.stats_dir = None
        a.tribes = tribes
        a.kind, a.opponent = kind, None
        return a

    def _record(self, args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.ladder.cmd_record(args)
        return json.loads(buf.getvalue())

    def _readings(self):
        with open(os.environ["LADDER_FILE"]) as f:
            return json.load(f)["readings"]

    def test_the_store_records_what_its_numbers_are_a_measurement_of(self):
        self._record(self._args())
        with open(os.environ["LADDER_FILE"]) as f:
            self.assertIn("Imperius", json.load(f)["scope"])

    def test_a_legacy_ladder_gains_the_scope_note_on_its_next_write(self):
        with open(os.environ["LADDER_FILE"], "w") as f:
            json.dump({"anchors": [{"name": "greedy", "path": "", "elo": 0.0}],
                       "readings": []}, f)
        self._record(self._args())
        with open(os.environ["LADDER_FILE"]) as f:
            self.assertEqual(json.load(f)["scope"], self.ladder.SCOPE_NOTE)

    def test_a_tribe_audit_reads_against_the_same_anchor_as_the_gauge(self):
        gauge = self._record(self._args())
        audit = self._record(self._args(kind="tribe_audit", tribes="Bardur,Kickoo"))
        self.assertEqual(audit["opponent"], gauge["opponent"])
        self.assertEqual(self._readings()[-1]["tribes"], "Bardur,Kickoo")

    def test_a_tribe_audit_carries_no_verdict_and_no_strike(self):
        for i in range(self.ladder.PLATEAU_WINDOW * 2):
            self._record(self._args(kind="tribe_audit", iteration=i,
                                    tribes="Bardur,Kickoo"))
        verdict = self._record(self._args(kind="tribe_audit", iteration=99,
                                          tribes="Bardur,Kickoo"))
        self.assertEqual(verdict["action"], "continue")
        self.assertEqual(verdict["plateau_strikes"], 0)

    def test_a_tribe_audit_stays_out_of_the_plateau_window(self):
        for i in range(self.ladder.PLATEAU_WINDOW):
            self._record(self._args(kind="tribe_audit", iteration=i,
                                    tribes="Bardur,Kickoo"))
        self.assertEqual(self.ladder._gauge_series(self.ladder._load()), [])

    def test_a_tribe_audit_stays_out_of_the_elo_fit(self):
        """Its games share the (model, anchor) node pair with the pinned
        reading, so pooling them would refold the block effect into the Elo."""
        self._record(self._args())
        pinned = elo_module().load_ladder_games(os.environ["LADDER_FILE"])
        self._record(self._args(kind="tribe_audit", tribes="Bardur,Kickoo"))
        self.assertEqual(
            elo_module().load_ladder_games(os.environ["LADDER_FILE"]), pinned
        )


def elo_module():
    import elo

    return elo


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP_SCRIPT = os.path.join(ROOT, "run_training_loop.sh")

# One representative value per flag run_training_loop.sh passes to ladder.py.
# A flag the loop gains and this table has not is a hard failure, not a skip:
# the point of these tests is that the command lines the loop builds are the
# ones that get run.
SHELL_FLAG_VALUES = {
    "--run-id": "1755000000",
    "--iteration": "7",
    "--wins": "13",
    "--losses": "3",
    "--draws": "0",
    "--avg-score-model": "1200.5",
    "--avg-score-opponent": "900.25",
    "--mcts": "64",
    "--gumbel-k": "16",
    "--eval-backend": "candle",
    "--wins-p1": "7",
    "--wins-p2": "6",
    "--games-attempted": "16",
    "--games-dropped": "0",
    "--unpaired-seeds": "0",
    "--tribes": "Imperius,Imperius",
    "--max-turns": "45",
    "--prior-heuristic-w": "0.5",
    "--q-weight": "0.0",
    "--stats-dir": "",
    "--path": "checkpoints/anchor_iter7_20260822_010203.safetensors",
    "--kind": "gauge",
    "--opponent": None,
}


def _contract_module():
    """scripts/check_cli_contract.py, which already knows how to read a shell
    script's flags. Importing it keeps these command lines tied to the loop
    instead of to a copy of it that can drift."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "check_cli_contract.py")
    spec = importlib.util.spec_from_file_location("check_cli_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShellCommandLineTest(unittest.TestCase):
    """#35: `freeze` and `audit-opponents` are reached from run_training_loop.sh
    and nowhere else, on a branch no smoke, test or CI check had ever entered —
    so the first execution of this shell<->argparse contract would have been
    mid-campaign, in a loop that aborts on a failed reading. These run the
    command lines the loop builds, with the flags read back off the loop."""

    @classmethod
    def setUpClass(cls):
        contract = _contract_module()
        with open(LOOP_SCRIPT, encoding="utf-8") as fh:
            lines = contract.logical_lines(fh.read())
        _, flag_vars = contract.collect_assignments(lines, contract.known_binaries())
        usage = contract.collect_python_usage(lines, flag_vars, LOOP_SCRIPT)
        cls.shell_usage = {
            target[len("ladder.py "):]: flags
            for target, flags in usage.items()
            if target.startswith("ladder.py ")
        }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ladder_file = os.path.join(self.tmp.name, "ladder.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _flags(self, subcommand):
        self.assertIn(subcommand, self.shell_usage,
                      f"run_training_loop.sh no longer invokes `ladder.py {subcommand}`")
        return self.shell_usage[subcommand]

    def _run(self, subcommand, overrides=None, env=None, check=True):
        """Invoke ladder.py exactly as the loop does: every flag the loop passes
        to this subcommand, with a representative value."""
        import subprocess

        values = dict(SHELL_FLAG_VALUES, **(overrides or {}))
        argv = [sys.executable, os.path.join(ROOT, "ladder.py"), subcommand]
        for flag in sorted(self._flags(subcommand)):
            self.assertIn(flag, values,
                          f"run_training_loop.sh passes {flag} to `{subcommand}` "
                          "but this test has no value for it")
            if values[flag] is not None:
                argv += [flag, values[flag]]
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            env=dict(os.environ, LADDER_FILE=self.ladder_file, **(env or {})),
        )
        if check:
            self.assertEqual(proc.returncode, 0,
                             f"{' '.join(argv[1:])}\n{proc.stderr}")
        return proc

    def _ladder(self):
        with open(self.ladder_file) as f:
            return json.load(f)

    def _freeze(self, path=None, iteration="7"):
        overrides = {"--iteration": iteration}
        if path is not None:
            overrides["--path"] = path
        return self._run("freeze", overrides)

    def test_the_loop_s_freeze_command_line_registers_an_anchor(self):
        anchor = self._freeze()
        registered = json.loads(anchor.stdout)
        data = self._ladder()
        self.assertEqual(data["anchors"][-1], registered)
        self.assertEqual(registered["path"], SHELL_FLAG_VALUES["--path"])
        self.assertEqual(registered["name"], "anchor_iter7_20260822_010203")
        self.assertEqual(registered["frozen_iteration"], 7)
        # The link match is what puts the new anchor on the outgoing one's Elo
        # scale, so it has to land as a reading, against the outgoing anchor.
        link = [r for r in data["readings"] if r["kind"] == "link"]
        self.assertEqual(len(link), 1)
        self.assertEqual(link[0]["opponent"], "greedy")
        self.assertEqual(link[0]["tribes"], SHELL_FLAG_VALUES["--tribes"])

    def test_a_freeze_clears_the_plateau_strikes_of_the_run_it_names(self):
        self._freeze()
        data = self._ladder()
        self.assertEqual(data["plateau_strikes"], 0)
        self.assertEqual(data["plateau_run_id"], SHELL_FLAG_VALUES["--run-id"])

    def test_audit_opponents_emits_what_the_loop_pipes_into_json_array_items(self):
        # Before any freeze the active anchor is greedy and there is nothing to
        # cross-check against, so the loop's while-read body must run zero times.
        self.assertEqual(json.loads(self._run("audit-opponents").stdout), [])

        self._freeze()
        opponents = json.loads(self._run("audit-opponents").stdout)
        self.assertEqual([o["name"] for o in opponents], ["greedy"])
        # `json_get name` / `json_get path` are what the loop reads off each item.
        for opponent in opponents:
            self.assertIn("name", opponent)
            self.assertIn("path", opponent)

    def test_audit_opponents_rotates_through_the_retired_net_anchors(self):
        self._freeze(path="checkpoints/anchor_iter7_a.safetensors")
        self._freeze(path="checkpoints/anchor_iter8_b.safetensors", iteration="8")
        names = [o["name"] for o in json.loads(self._run("audit-opponents").stdout)]
        self.assertEqual(names, ["greedy", "anchor_iter7_a"])

    def test_the_loop_s_audit_command_line_records_a_cross_check_row(self):
        self._freeze()
        verdict = json.loads(
            self._run("record", {"--kind": "audit", "--opponent": "greedy"}).stdout
        )
        self.assertEqual(verdict["action"], "continue")
        self.assertEqual(verdict["opponent"], "greedy")
        row = self._ladder()["readings"][-1]
        self.assertEqual(row["kind"], "audit")
        self.assertEqual(row["opponent"], "greedy")

    def test_an_audit_row_needs_the_anchor_the_loop_read_off_audit_opponents(self):
        proc = self._run("record", {"--kind": "audit", "--opponent": None}, check=False)
        self.assertNotEqual(proc.returncode, 0)

    def test_a_tribe_audit_names_the_anchor_its_match_was_played_against(self):
        """The audit cadence can land on a freeze iteration, and the loop plays
        every cross-check of that iteration against the anchor the gauge used —
        which the freeze has by then retired."""
        self._freeze()
        outgoing = "greedy"
        self._run("record", {"--kind": "tribe_audit", "--opponent": outgoing,
                             "--tribes": "Bardur,Kickoo"})
        row = self._ladder()["readings"][-1]
        self.assertEqual(row["kind"], "tribe_audit")
        self.assertEqual(row["opponent"], outgoing)
        self.assertEqual(row["tribes"], "Bardur,Kickoo")

    def test_a_tribe_audit_without_an_opponent_still_reads_the_active_anchor(self):
        self._run("record", {"--kind": "tribe_audit", "--opponent": None,
                             "--tribes": "Bardur,Kickoo"})
        self.assertEqual(self._ladder()["readings"][-1]["opponent"], "greedy")

    def test_the_gauge_command_line_answers_with_a_verdict_the_loop_can_read(self):
        verdict = json.loads(self._run("record", {"--kind": "gauge"}).stdout)
        # json_get action / underpowered_for_pp / resolves_pp / games_needed.
        self.assertIn(verdict["action"], ("continue", "freeze", "stop"))
        self.assertIn("resolves_pp", verdict)
        self.assertNotIn("freeze_wr", verdict)

    def test_a_lowered_freeze_bar_reaches_the_branch_and_says_so(self):
        """What the smoke leans on: at the production bar no reading it can
        afford reaches the freeze branch, so the bar is lowered there — and a
        forced freeze must never be indistinguishable from an earned one."""
        verdict = json.loads(
            self._run("record", {"--kind": "gauge", "--wins": "0", "--losses": "16"},
                      env={"GAUGE_FREEZE_WR": "0"}).stdout
        )
        self.assertEqual(verdict["action"], "freeze")
        self.assertEqual(verdict["freeze_wr"], 0.0)
        self.assertEqual(self._ladder()["readings"][-1]["freeze_wr"], 0.0)

    def test_the_production_bar_is_unmoved_by_the_hook(self):
        import ladder

        self.assertEqual(ladder.DEFAULT_FREEZE_WR, 0.80)
        verdict = json.loads(
            self._run("record", {"--kind": "gauge", "--wins": "0", "--losses": "16"}).stdout
        )
        self.assertEqual(verdict["action"], "continue")

    def test_every_flag_the_loop_passes_is_a_real_option(self):
        """The static half: the loop's flags checked against the parser itself,
        with no subprocess and no values. scripts/check_cli_contract.py runs the
        same check in CI over --help; this one fails in the fast test job too."""
        import ladder

        subparsers = ladder.build_parser()._subparsers._group_actions[0].choices
        self.assertTrue(self.shell_usage, "no ladder.py invocations found in the loop")
        for subcommand in ("active", "record", "freeze", "audit-opponents"):
            self.assertIn(subcommand, self.shell_usage,
                          f"run_training_loop.sh no longer invokes `ladder.py {subcommand}`")
        for subcommand, flags in sorted(self.shell_usage.items()):
            known = {opt for action in subparsers[subcommand]._actions
                     for opt in action.option_strings}
            self.assertEqual(sorted(flags - known), [],
                             f"`ladder.py {subcommand}` does not accept these")


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
