"""Tests for the power analysis.

The claim the module supports: the 43-scenario suite is underpowered for the
comparisons the paper wants. These pin the calculations that establish it.
"""

from __future__ import annotations

import math

from chokepoint.analysis import power


class TestRequiredN:
    def test_larger_effects_need_fewer_samples(self):
        big = power.required_n(0.60, 0.30)
        small = power.required_n(0.60, 0.50)
        assert big < small

    def test_zero_effect_is_undetectable(self):
        assert power.required_n(0.5, 0.5) is None

    def test_known_ballpark(self):
        # 0.60 vs 0.35 at 80% power needs on the order of 60 per group.
        n = power.required_n(0.60, 0.35, alpha=0.05, power=0.80)
        assert 45 <= n <= 80

    def test_higher_power_demands_more(self):
        assert power.required_n(0.6, 0.4, power=0.90) > power.required_n(0.6, 0.4, power=0.80)


class TestAchievedPower:
    def test_current_suite_is_underpowered_for_the_headline_comparison(self):
        """43 scenarios cannot reliably resolve 60% vs 35%."""
        assert power.achieved_power(0.60, 0.35, n=43) < 0.80

    def test_power_rises_with_n(self):
        assert power.achieved_power(0.6, 0.35, 90) > power.achieved_power(0.6, 0.35, 43)

    def test_identical_proportions_have_no_power(self):
        assert power.achieved_power(0.5, 0.5, 100) == 0.0

    def test_power_is_a_probability(self):
        for n in (10, 43, 200):
            assert 0.0 <= power.achieved_power(0.6, 0.3, n) <= 1.0


class TestMinimumDetectableEffect:
    def test_bigger_suites_detect_smaller_effects(self):
        small_n = power.minimum_detectable_effect(0.60, 43)
        large_n = power.minimum_detectable_effect(0.60, 200)
        assert large_n < small_n

    def test_returns_none_when_nothing_is_detectable(self):
        assert power.minimum_detectable_effect(0.60, 1) is None


class TestNormalQuantile:
    def test_matches_known_values(self):
        assert math.isclose(power._z(0.975), 1.96, abs_tol=0.01)
        assert math.isclose(power._z(0.95), 1.6449, abs_tol=0.01)

    def test_interpolates_uncommon_quantiles(self):
        # ~0.85 quantile is about 1.036
        assert math.isclose(power._z(0.85), 1.036, abs_tol=0.02)


class TestScan:
    def test_produces_a_row_per_size(self):
        rows = power.scan(0.6, [43, 90, 150])
        assert [r["n_per_group"] for r in rows] == [43, 90, 150]

    def test_mde_shrinks_as_n_grows(self):
        rows = power.scan(0.6, [43, 200])
        mdes = [r["minimum_detectable_reduction_pp"] for r in rows]
        assert mdes[1] < mdes[0]
