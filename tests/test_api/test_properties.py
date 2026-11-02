"""Hypothesis property tests: request/response JSON round-trips (spec §4).

Round-trip invariants: serializing a request or response to JSON and
validating it back is the identity — no field is dropped, defaults invented,
or provenance relabeled anywhere on the API boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from rechip01.disposition import optimize_disposition
from rechip01.passport import build_passport
from rechip01.passport.models import AssessRequest, ScanIngestRequest
from rechip01.schemas import Provenance
from tests.test_api.test_builder import make_estimate, make_scan
from tests.test_disposition.helpers import make_economy
from tests.test_disposition.strategies import economies
from tests.test_scanner.fixtures import valid_rows

_PROVENANCES = st.sampled_from(list(Provenance))
_RUN_IDS = st.from_regex(r"run-[a-z0-9]{4,12}", fullmatch=True)
_HORIZONS = st.integers(min_value=1, max_value=3650)
_WORKLOAD_SETS = st.sets(
    st.sampled_from(["inference", "finetune", "memory-stress"]), min_size=1, max_size=3
)


@settings(max_examples=50, deadline=None)
@given(
    provenance=_PROVENANCES,
    rows=st.lists(valid_rows(), max_size=4),
    source_format=st.sampled_from(["in_memory", "csv_export", "jsonl_export", "simulator_feed"]),
)
def test_scan_ingest_request_round_trip(
    provenance: Provenance, rows: list[dict[str, Any]], source_format: str
) -> None:
    """ScanIngestRequest survives a JSON round-trip unchanged."""
    request = ScanIngestRequest.model_validate(
        {"provenance": provenance, "rows": rows, "source_format": source_format}
    )
    revived = ScanIngestRequest.model_validate_json(request.model_dump_json())
    assert revived == request
    assert revived.provenance == request.provenance


@settings(max_examples=50, deadline=None)
@given(
    horizon_days=_HORIZONS,
    target_workloads=_WORKLOAD_SETS,
    economy=economies(),
    run_id=st.one_of(st.none(), _RUN_IDS),
)
def test_assess_request_round_trip(
    horizon_days: int,
    target_workloads: set[str],
    economy: Any,
    run_id: str | None,
) -> None:
    """AssessRequest survives a JSON round-trip unchanged, economy included."""
    request = AssessRequest(
        horizon_days=horizon_days,
        target_workloads=tuple(sorted(target_workloads)),
        economy=economy,
        asset_ids=None,
        run_id=run_id,
    )
    revived = AssessRequest.model_validate_json(request.model_dump_json())
    assert revived == request
    assert revived.economy == economy


@settings(max_examples=50, deadline=None)
@given(
    provenance=_PROVENANCES,
    p_fail=st.floats(0.01, 0.99, allow_nan=False, allow_infinity=False),
    run_id=_RUN_IDS,
)
def test_assess_response_passport_round_trip(
    provenance: Provenance, p_fail: float, run_id: str
) -> None:
    """A built passport survives a JSON round-trip: provenance, uncertainty
    bands, evidence trace, and disposition all preserved exactly."""
    scan = make_scan(provenance=provenance)
    estimate = make_estimate(
        asset_id=scan.asset_id, provenance=provenance, p_fail=p_fail, ci=(p_fail, p_fail)
    )
    # Deactivate the OOD review gate by matching the degenerate CI so the
    # optimizer recommends an economic path rather than manual_review.
    decision = optimize_disposition(estimate, make_economy(), seed=0)
    passport = build_passport(
        scan,
        estimate,
        decision,
        run_id=run_id,
        issued_at=datetime(2026, 11, 1, 12, 0, tzinfo=UTC),
    )

    revived = type(passport).model_validate_json(passport.model_dump_json())
    assert revived == passport
    assert revived.provenance == provenance
    assert revived.reliability.ci == passport.reliability.ci
    assert revived.reliability.p_fail == passport.reliability.p_fail
    assert revived.evidence_trace == passport.evidence_trace
    assert revived.disposition.recommendation == decision.recommendation
