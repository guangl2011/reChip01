"""Shared fixtures for the simulator test suite."""

from typing import Any

import pytest

from rechip01.simulator import FleetConfig, default_fleet_config

# Deterministic scenario configs: n_units large enough that aggregate
# assertions (monotonicity, calibration) hold with wide margins, small enough
# that the suite stays fast.
SCENARIO_UNITS = 4000


@pytest.fixture
def scenario_config() -> FleetConfig:
    """The default built-in fleet at scenario scale."""
    return default_fleet_config(seed=1234, n_units=SCENARIO_UNITS)


@pytest.fixture
def with_multipliers() -> Any:
    """Return a callable that overrides top-level multipliers on a config."""

    def _with_multipliers(config: FleetConfig, **overrides: float) -> FleetConfig:
        return config.model_copy(update=overrides)

    return _with_multipliers
