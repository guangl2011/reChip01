"""Contract tests for disposition economy/config validation (spec art_vuIPvUhR, §4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rechip01.disposition import DispositionConfig, PriceRange
from tests.test_disposition.helpers import make_economy


class TestPriceRange:
    def test_midpoint(self) -> None:
        assert PriceRange(low=100.0, high=200.0).midpoint() == 150.0

    def test_zero_width_allowed(self) -> None:
        assert PriceRange(low=100.0, high=100.0).midpoint() == 100.0

    def test_low_above_high_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PriceRange(low=200.0, high=100.0)

    def test_negative_low_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PriceRange(low=-1.0, high=100.0)


class TestEconomyAssumptions:
    def test_frozen_after_construction(self) -> None:
        economy = make_economy()
        with pytest.raises((TypeError, ValidationError)):
            economy.test_cost = 1.0

    def test_empty_resale_prices_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_economy(resale_price_by_workload={})

    def test_negative_test_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_economy(test_cost=-0.01)

    def test_zero_test_cost_allowed(self) -> None:
        assert make_economy(test_cost=0.0).test_cost == 0.0

    def test_residual_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_economy(repair_residual_p_fail=1.5)

    def test_negative_repair_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_economy(repair_cost=-1.0)

    def test_negative_recycling_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_economy(recycling_value=(-5.0, 10.0))


class TestDispositionConfig:
    def test_defaults(self) -> None:
        config = DispositionConfig()
        assert config.max_ci_width == 0.35
        assert config.review_cost == 25.0
        assert config.n_samples == 2048

    def test_max_ci_width_is_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            DispositionConfig(max_ci_width=1.5)
        assert DispositionConfig(max_ci_width=0.0).max_ci_width == 0.0

    def test_n_samples_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            DispositionConfig(n_samples=0)
        assert DispositionConfig(n_samples=1).n_samples == 1

    def test_negative_review_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DispositionConfig(review_cost=-0.01)
