"""Guardrail tests: if the data or the logic moves, these fail loudly.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insight import analysis as a  # noqa: E402
from insight import charts  # noqa: E402
from insight import metrics as m  # noqa: E402
from insight.data import load  # noqa: E402


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw, cls.report = load()
        cls.df = m.clean(cls.raw)

    def test_raw_row_count(self) -> None:
        self.assertEqual(self.report.rows, 1277)

    def test_missing_outcomes_are_excluded(self) -> None:
        self.assertEqual(self.report.missing_assembled, 49)
        self.assertFalse(self.df["assembled"].isna().any())

    def test_exact_duplicates_removed(self) -> None:
        self.assertEqual(len(self.df), 1169)
        cols = [c for c in self.raw.columns if c not in ("delay_min",)]
        self.assertEqual(len(self.df.drop_duplicates(subset=cols)), len(self.df))

    def test_outcome_counts(self) -> None:
        self.assertEqual(int((self.df["assembled"] == 0).sum()), 141)
        self.assertAlmostEqual(float((self.df["assembled"] == 0).mean()), 0.1206, places=3)

    def test_delay_arithmetic(self) -> None:
        row = self.df.iloc[0]
        expected = (row["request_time"] - row["stock_synced_at"]).total_seconds() / 60
        self.assertAlmostEqual(float(row["delay_min"]), expected, places=6)
        self.assertGreaterEqual(float(self.df["delay_min"].min()), 0.0)

    def test_timezone_alignment(self) -> None:
        # request_time 07:19+08:00 == 23:19Z; lag must be 1 min, not 481
        first = self.raw.iloc[0]
        self.assertAlmostEqual(float(first["delay_min"]), 1.0, places=6)


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = m.clean(load()[0])
        cls.curve = m.delay_bucket_curve(cls.df).set_index("delay_bucket")

    def test_below_threshold_is_flat(self) -> None:
        for bucket in ("0-5", "5-10", "10-15", "15-20"):
            self.assertLess(float(self.curve.loc[bucket, "missing_rate"]), 0.05)

    def test_above_threshold_jumps(self) -> None:
        for bucket in ("20-30", "30-45", "45-60", "60-71"):
            self.assertGreater(float(self.curve.loc[bucket, "missing_rate"]), 0.25)

    def test_no_overlap_between_bands(self) -> None:
        low_hi = max(float(self.curve.loc[b, "ci_high"]) for b in ("0-5", "5-10", "10-15", "15-20"))
        high_lo = min(float(self.curve.loc[b, "ci_low"]) for b in ("20-30", "30-45", "45-60", "60-71"))
        self.assertLess(low_hi, high_lo)


class RuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = m.clean(load()[0])
        cls.sweep = m.threshold_sweep(cls.df)
        cls.best = m.optimal_threshold(cls.sweep)

    def test_optimum_is_19(self) -> None:
        self.assertEqual(self.best["threshold"], 19.0)

    def test_optimum_metrics(self) -> None:
        self.assertEqual(self.best["intercepted"], 323)
        self.assertEqual(self.best["true_hits"], 110)
        self.assertAlmostEqual(self.best["hit_rate"], 0.3406, places=3)
        self.assertAlmostEqual(self.best["net_savings_min"], 4374.0, places=0)
        self.assertAlmostEqual(self.best["savings_vs_baseline"], 0.4136, places=3)

    def test_net_saving_formula(self) -> None:
        r = m.evaluate_threshold(self.df, 19.0)
        self.assertAlmostEqual(
            r.net_savings_min,
            r.true_hits * m.BASELINE_COST_MIN - r.intercepted * m.INTERCEPT_COST_MIN,
        )

    def test_baseline_matches_stoppage_count(self) -> None:
        self.assertAlmostEqual(
            int((self.df["assembled"] == 0).sum()) * m.BASELINE_COST_MIN, 10575.0
        )

    def test_breakeven_is_below_actual_hit_rate(self) -> None:
        self.assertLess(self.best["breakeven_hit_rate"], self.best["hit_rate"])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = m.clean(load()[0])

    def test_two_group_difference_is_significant(self) -> None:
        r = m.two_group_test(self.df, 20)
        self.assertLess(r["p_value"], 1e-20)
        self.assertGreater(r["odds_ratio"], 5.0)
        self.assertGreater(r["rate_ratio"], 5.0)

    def test_lag_is_the_strongest_predictor(self) -> None:
        preds = a.predictor_comparison(self.df).set_index("predictor")
        self.assertGreater(float(preds.loc["delay_min", "best_auc"]), 0.75)
        self.assertGreater(
            float(preds.loc["delay_min", "best_auc"]),
            float(preds.loc["recorded_stock", "best_auc"]),
        )
        self.assertGreater(
            float(preds.loc["delay_min", "best_auc"]),
            float(preds.loc["queue_length", "best_auc"]),
        )

    def test_workshop_difference_exists(self) -> None:
        t = a.segment_test(self.df, "workshop")
        self.assertLess(t["p_value"], 0.05)

    def test_category_difference_is_not_significant(self) -> None:
        t = a.segment_test(self.df, "part_category")
        self.assertGreater(t["p_value"], 0.05)

    def test_holdout_generalises(self) -> None:
        h = a.holdout_check(self.df)
        self.assertEqual(h["chosen_threshold"], 19.0)
        self.assertGreater(h["test_hit_rate"], 0.20)
        self.assertGreater(h["test_net_savings_min"], 0.0)

    def test_adding_stock_does_not_help(self) -> None:
        """The stock rule must not beat lag alone (it ties or loses money)."""
        abl = a.ablation(self.df).set_index("rule")
        self.assertLessEqual(
            float(abl.loc["lag >= T OR stock <= S", "net_savings_min"]),
            float(abl.loc["ledger lag >= T", "net_savings_min"]),
        )


class ChartTests(unittest.TestCase):
    def test_svgs_are_well_formed(self) -> None:
        for name in ("fig1_delay_curve", "fig2_threshold_sweep", "fig3_workshop"):
            path = f"outputs/figures/{name}.svg"
            self.assertTrue(os.path.exists(path), f"missing {path}; run scripts/run_analysis.py")
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag.split("}")[-1], "svg")
            self.assertGreater(len(list(root)), 10)

    def test_bar_chart_numbers_match_input(self) -> None:
        svg = charts.bar_chart(["a", "b"], [0.25, 0.5], [(0.1, 0.3), (0.4, 0.6)])
        self.assertIn("25.0%", svg)
        self.assertIn("50.0%", svg)
        self.assertEqual(svg.count("</svg>"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
