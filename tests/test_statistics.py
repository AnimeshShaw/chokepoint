"""Tests for the rewritten statistics module.

The pre-audit analysis reconstructed counts from rounded percentages and never
computed a p-value. These pin the exact tests that replaced it.
"""

from __future__ import annotations

import math

from chokepoint.analysis import statistics as stats


class TestWilsonInterval:
    def test_interval_brackets_the_point_estimate(self):
        p, lo, hi = stats.wilson_interval(27, 43)
        assert lo < p < hi

    def test_zero_successes_gives_zero_lower_bound(self):
        p, lo, hi = stats.wilson_interval(0, 43)
        assert p == 0.0
        assert lo == 0.0
        assert hi > 0.0

    def test_known_value(self):
        p, lo, hi = stats.wilson_interval(50, 100)
        assert p == 50.0
        assert math.isclose(lo, 40.38, abs_tol=0.1)
        assert math.isclose(hi, 59.62, abs_tol=0.1)


class TestFisherExact:
    def test_identical_proportions_are_not_significant(self):
        assert stats.fisher_exact_two_sided(10, 10, 10, 10) > 0.9

    def test_strong_separation_is_significant(self):
        assert stats.fisher_exact_two_sided(27, 16, 14, 29) < 0.05

    def test_p_value_is_a_probability(self):
        for table in [(1, 9, 8, 2), (0, 43, 27, 16), (5, 5, 5, 5)]:
            p = stats.fisher_exact_two_sided(*table)
            assert 0.0 <= p <= 1.0

    def test_matches_scipy_when_available(self):
        try:
            from scipy.stats import fisher_exact
        except ImportError:
            return
        for table in [(27, 16, 14, 29), (3, 40, 1, 42)]:
            expected = fisher_exact([[table[0], table[1]], [table[2], table[3]]])[1]
            assert math.isclose(stats.fisher_exact_two_sided(*table), expected, rel_tol=1e-6)


class TestHolmBonferroni:
    def test_correction_never_lowers_a_p_value(self):
        raw = {"a": 0.01, "b": 0.04, "c": 0.20}
        out = stats.holm_bonferroni(raw)
        for key, value in out.items():
            assert value["p_adjusted"] >= raw[key]

    def test_adjusted_values_are_monotone(self):
        out = stats.holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03})
        adjusted = [out[k]["p_adjusted"] for k in ("a", "b", "c")]
        assert adjusted == sorted(adjusted)

    def test_marginal_result_can_lose_significance_after_correction(self):
        """The reason correction matters for a multi-defense comparison."""
        out = stats.holm_bonferroni({"d1": 0.03, "d2": 0.04, "d3": 0.045})
        assert not out["d3"]["significant"]


class TestCohensH:
    def test_zero_for_identical_proportions(self):
        assert stats.cohens_h(0.5, 0.5) == 0.0

    def test_symmetric(self):
        assert stats.cohens_h(0.2, 0.6) == stats.cohens_h(0.6, 0.2)
