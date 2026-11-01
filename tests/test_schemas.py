"""Unit and property tests for the core schema contracts (spec §4)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from rechip01.schemas import (
    AssetPassport,
    DispositionDecision,
    OptionEvaluation,
    Provenance,
    ReliabilityEstimate,
    ScanResult,
)

# --- valid-fixture builders (kwargs form so tests can drop/override fields) ---


def scan_result_kwargs() -> dict[str, Any]:
    """Return a valid ScanResult kwargs dict."""
    return {
        "asset_id": "GPU-0001",
        "model": "A100-SXM4-80GB",
        "provenance": Provenance.MEASURED,
        "age_months": 30.0,
        "power_on_hours": 18_000.0,
        "ecc_events": 4,
        "xid_errors": 0,
        "retired_pages": 2,
        "thermal_history": [0.42, 0.55, 0.61],
        "max_sustained_power_pct": 0.785,
        "bench_mark_snapshot": {"dcgm_mem_util": 0.91, "dcgm_diag_result": 1.0},
        "chain_of_custody": ["datacenter-a intake", "erasure per NIST 800-88"],
    }


def reliability_estimate_kwargs() -> dict[str, Any]:
    """Return a valid ReliabilityEstimate kwargs dict."""
    return {
        "asset_id": "GPU-0001",
        "horizon_days": 90,
        "provenance": Provenance.MEASURED,
        "p_fail": 0.08,
        "ci": (0.03, 0.15),
        "workload_fitness": {"inference": 0.83, "finetune": 0.61},
        "ood_flag": False,
        "model_version": "reliability-0.1.0",
    }


def disposition_decision_kwargs() -> dict[str, Any]:
    """Return a valid DispositionDecision kwargs dict."""
    return {
        "asset_id": "GPU-0001",
        "provenance": Provenance.MEASURED,
        "recommendation": "reuse_inference",
        "expected_net_recovery": 412.5,
        "regret_bound": 55.0,
        "options_evaluated": [
            OptionEvaluation(
                path="reuse_inference", expected_net_recovery=412.5, regret_bound=55.0
            ),
            OptionEvaluation(path="harvest", expected_net_recovery=210.0, regret_bound=10.0),
        ],
        "assumption_trace": ["reuse price at 0.55 x MSRP", "harvest yield 0.8"],
        "model_version": "disposition-0.1.0",
    }


def passport_kwargs() -> dict[str, Any]:
    """Return a valid AssetPassport kwargs dict."""
    return {
        "asset_id": "GPU-0001",
        "run_id": "run-2026-11-01-001",
        "provenance": Provenance.MEASURED,
        "issued_at": datetime(2026, 11, 1, 12, 0, tzinfo=UTC),
        "reliability": ReliabilityEstimate(**reliability_estimate_kwargs()),
        "disposition": DispositionDecision(**disposition_decision_kwargs()),
        "evidence_trace": ["scan:batch-7", "model:reliability-0.1.0"],
        "model_version": "reliability-0.1.0",
    }


# --- hypothesis strategies over valid contract instances ---


def unit_floats() -> st.SearchStrategy[float]:
    """Finite floats in [0, 1]."""
    return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def scan_results(draw: st.DrawFn) -> ScanResult:
    """Arbitrary valid ScanResult instances."""
    kwargs = scan_result_kwargs()
    kwargs["provenance"] = draw(st.sampled_from(list(Provenance)))
    kwargs["age_months"] = draw(
        st.one_of(st.none(), st.floats(min_value=0.0, max_value=120.0, allow_nan=False))
    )
    kwargs["power_on_hours"] = draw(
        st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e5, allow_nan=False))
    )
    kwargs["ecc_events"] = draw(st.integers(min_value=0, max_value=10**6))
    kwargs["xid_errors"] = draw(st.integers(min_value=0, max_value=10**3))
    kwargs["retired_pages"] = draw(st.integers(min_value=0, max_value=10**5))
    kwargs["thermal_history"] = draw(st.lists(unit_floats(), max_size=8))
    kwargs["max_sustained_power_pct"] = draw(unit_floats())
    kwargs["bench_mark_snapshot"] = draw(
        st.dictionaries(st.text(min_size=1, max_size=12), unit_floats(), max_size=4)
    )
    return ScanResult(**kwargs)


@st.composite
def reliability_estimates(draw: st.DrawFn) -> ReliabilityEstimate:
    """Arbitrary valid ReliabilityEstimate instances (ci brackets p_fail)."""
    kwargs = reliability_estimate_kwargs()
    kwargs["provenance"] = draw(st.sampled_from(list(Provenance)))
    kwargs["horizon_days"] = draw(st.integers(min_value=1, max_value=3_650))
    p_fail = draw(unit_floats())
    low = draw(st.floats(min_value=0.0, max_value=p_fail, allow_nan=False))
    high = draw(st.floats(min_value=p_fail, max_value=1.0, allow_nan=False))
    kwargs["p_fail"] = p_fail
    kwargs["ci"] = (low, high)
    kwargs["workload_fitness"] = draw(
        st.dictionaries(st.text(min_size=1, max_size=12), unit_floats(), max_size=4)
    )
    kwargs["ood_flag"] = draw(st.booleans())
    return ReliabilityEstimate(**kwargs)


@st.composite
def disposition_decisions(draw: st.DrawFn) -> DispositionDecision:
    """Arbitrary valid DispositionDecision instances."""
    kwargs = disposition_decision_kwargs()
    kwargs["provenance"] = draw(st.sampled_from(list(Provenance)))
    paths = draw(st.lists(st.text(min_size=1, max_size=16), min_size=1, max_size=4, unique=True))
    options = [
        OptionEvaluation(
            path=path,
            expected_net_recovery=draw(
                st.floats(min_value=-10_000.0, max_value=10_000.0, allow_nan=False)
            ),
            regret_bound=draw(st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False)),
        )
        for path in paths
    ]
    kwargs["options_evaluated"] = options
    kwargs["recommendation"] = draw(st.sampled_from(paths))
    kwargs["expected_net_recovery"] = next(
        option.expected_net_recovery
        for option in options
        if option.path == kwargs["recommendation"]
    )
    kwargs["regret_bound"] = next(
        option.regret_bound for option in options if option.path == kwargs["recommendation"]
    )
    return DispositionDecision(**kwargs)


# --- serialization round-trips (property tests) ---


@given(data=scan_results())
def test_scan_result_json_round_trip(data: ScanResult) -> None:
    """Any valid ScanResult survives a JSON serialization round-trip exactly."""
    assert ScanResult.model_validate_json(data.model_dump_json()) == data


@given(data=reliability_estimates())
def test_reliability_estimate_json_round_trip(data: ReliabilityEstimate) -> None:
    """Any valid ReliabilityEstimate survives a JSON round-trip exactly."""
    assert ReliabilityEstimate.model_validate_json(data.model_dump_json()) == data


@given(data=disposition_decisions())
def test_disposition_decision_json_round_trip(data: DispositionDecision) -> None:
    """Any valid DispositionDecision survives a JSON round-trip exactly."""
    assert DispositionDecision.model_validate_json(data.model_dump_json()) == data


@given(
    estimate=reliability_estimates(),
    decision=disposition_decisions(),
    provenance=st.sampled_from(list(Provenance)),
)
def test_passport_json_round_trip(
    estimate: ReliabilityEstimate,
    decision: DispositionDecision,
    provenance: Provenance,
) -> None:
    """Any valid AssetPassport survives a JSON round-trip exactly."""
    kwargs = passport_kwargs()
    kwargs["provenance"] = provenance
    kwargs["reliability"] = estimate
    kwargs["disposition"] = decision
    passport = AssetPassport(**kwargs)
    assert AssetPassport.model_validate_json(passport.model_dump_json()) == passport


# --- provenance is required on every contract that carries numbers ---


@pytest.mark.parametrize(
    ("model_cls", "kwargs_fn"),
    [
        (ScanResult, scan_result_kwargs),
        (ReliabilityEstimate, reliability_estimate_kwargs),
        (DispositionDecision, disposition_decision_kwargs),
        (AssetPassport, passport_kwargs),
    ],
)
def test_provenance_is_required(model_cls: type, kwargs_fn: Any) -> None:
    """Dropping provenance from any contract is a validation error, never a default."""
    kwargs = {key: value for key, value in kwargs_fn().items() if key != "provenance"}
    with pytest.raises(ValidationError, match="provenance"):
        model_cls(**kwargs)


def test_provenance_values_are_simulated_or_measured() -> None:
    """No third data class exists: provenance is simulated or measured, period."""
    assert {member.value for member in Provenance} == {"simulated", "measured"}


# --- immutability: provenance stamped once, never mutated downstream ---


@pytest.mark.parametrize(
    ("model_cls", "kwargs_fn"),
    [
        (ScanResult, scan_result_kwargs),
        (ReliabilityEstimate, reliability_estimate_kwargs),
        (DispositionDecision, disposition_decision_kwargs),
        (AssetPassport, passport_kwargs),
    ],
)
def test_contracts_are_frozen(model_cls: type, kwargs_fn: Any) -> None:
    """Every contract rejects mutation after construction."""
    instance = model_cls(**kwargs_fn())
    with pytest.raises(ValidationError):
        instance.provenance = Provenance.SIMULATED


# --- validation rejections for out-of-range and inconsistent values ---


def test_scan_result_rejects_out_of_range_values() -> None:
    """Out-of-range telemetry and empty custody chains are rejected."""
    for field, value in [
        ("age_months", -1.0),
        ("power_on_hours", -0.5),
        ("ecc_events", -1),
        ("xid_errors", -1),
        ("retired_pages", -1),
        ("max_sustained_power_pct", 1.5),
    ]:
        with pytest.raises(ValidationError):
            ScanResult(**{**scan_result_kwargs(), field: value})
    with pytest.raises(ValidationError):
        ScanResult(**{**scan_result_kwargs(), "chain_of_custody": []})


def test_scan_result_rejects_non_finite_numbers() -> None:
    """NaN and infinity never enter the pipeline through telemetry."""
    kwargs = scan_result_kwargs()
    kwargs["thermal_history"] = [float("nan")]
    with pytest.raises(ValidationError):
        ScanResult(**kwargs)


def test_reliability_estimate_rejects_out_of_range_p_fail() -> None:
    """p_fail outside [0, 1] is rejected."""
    kwargs = reliability_estimate_kwargs()
    kwargs["p_fail"] = 1.5
    with pytest.raises(ValidationError):
        ReliabilityEstimate(**kwargs)


def test_reliability_estimate_rejects_zero_horizon() -> None:
    """A non-positive failure horizon is meaningless and rejected."""
    kwargs = reliability_estimate_kwargs()
    kwargs["horizon_days"] = 0
    with pytest.raises(ValidationError):
        ReliabilityEstimate(**kwargs)


@given(
    p_fail=unit_floats(),
    ci_low=unit_floats(),
    ci_high=unit_floats(),
)
def test_ci_not_bracketing_p_fail_is_rejected(p_fail: float, ci_low: float, ci_high: float) -> None:
    """Any interval that does not bracket p_fail is rejected; valid ones pass."""
    kwargs = reliability_estimate_kwargs()
    kwargs["p_fail"] = p_fail
    kwargs["ci"] = (ci_low, ci_high)
    if ci_low <= p_fail <= ci_high:
        assert ReliabilityEstimate(**kwargs).p_fail == p_fail
    else:
        with pytest.raises(ValidationError, match="bracket"):
            ReliabilityEstimate(**kwargs)


def test_workload_fitness_rejects_out_of_range_scores() -> None:
    """Fitness scores are unit-interval values."""
    kwargs = reliability_estimate_kwargs()
    kwargs["workload_fitness"] = {"inference": 1.2}
    with pytest.raises(ValidationError):
        ReliabilityEstimate(**kwargs)


def test_disposition_rejects_unrecommended_option() -> None:
    """The recommendation must be among the evaluated options."""
    kwargs = disposition_decision_kwargs()
    kwargs["recommendation"] = "recycle"
    with pytest.raises(ValidationError, match="evaluated"):
        DispositionDecision(**kwargs)


def test_disposition_rejects_negative_regret_bound() -> None:
    """A negative regret bound is nonsensical and rejected."""
    kwargs = disposition_decision_kwargs()
    kwargs["regret_bound"] = -0.01
    with pytest.raises(ValidationError):
        DispositionDecision(**kwargs)


def test_passport_rejects_naive_issued_at() -> None:
    """issued_at must be timezone-aware so passport ordering is unambiguous."""
    kwargs = passport_kwargs()
    kwargs["issued_at"] = datetime(2026, 11, 1, 12, 0)
    with pytest.raises(ValidationError, match="timezone"):
        AssetPassport(**kwargs)
