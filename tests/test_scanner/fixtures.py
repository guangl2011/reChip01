"""Shared row builders and Hypothesis strategies for scanner tests."""

from __future__ import annotations

import json
from typing import Final

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

ASSET_ID_PATTERN: Final[str] = r"[A-Za-z0-9][A-Za-z0-9_-]{0,11}"


def valid_row(**overrides: object) -> dict[str, object]:
    """A fully populated, schema-valid row; ``overrides`` replace fields."""
    row: dict[str, object] = {
        "asset_id": "GPU-0001",
        "model": "A100-SXM4-80GB",
        "age_months": 30.0,
        "power_on_hours": 18250.0,
        "ecc_events": 12,
        "xid_errors": 1,
        "retired_pages": 3,
        "thermal_history": [0.61, 0.72, 0.68],
        "max_sustained_power_pct": 0.87,
        "bench_mark_snapshot": {"pcie_replay_rate": 0.01},
        "chain_of_custody": ["intake", "erasure"],
    }
    row.update(overrides)
    return row


def stringify_row(row: dict[str, object]) -> dict[str, str]:
    """Encode a typed row the way a CSV export would carry it (all strings)."""
    encoded: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            encoded[key] = json.dumps(value)
        else:
            encoded[key] = str(value)
    return encoded


_asset_ids = st.from_regex(ASSET_ID_PATTERN, fullmatch=True)
_counters = st.integers(min_value=0, max_value=1_000_000)
_unit_floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_nonneg_floats = st.floats(
    min_value=0.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
)
_metrics = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def valid_rows(draw: DrawFn) -> dict[str, object]:
    """Arbitrary schema-valid rows (all fields present, all values in range)."""
    return {
        "asset_id": draw(_asset_ids),
        "model": draw(st.from_regex(r"[A-Za-z0-9-]{3,20}", fullmatch=True)),
        "age_months": draw(_nonneg_floats),
        "power_on_hours": draw(_nonneg_floats),
        "ecc_events": draw(_counters),
        "xid_errors": draw(_counters),
        "retired_pages": draw(_counters),
        "thermal_history": draw(st.lists(_unit_floats, max_size=8)),
        "max_sustained_power_pct": draw(_unit_floats),
        "bench_mark_snapshot": draw(
            st.dictionaries(
                st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8),
                _metrics,
                max_size=4,
            )
        ),
        "chain_of_custody": draw(st.lists(_asset_ids, min_size=1, max_size=4)),
    }


@st.composite
def corrupted_rows(draw: DrawFn) -> dict[str, object]:
    """A valid row with exactly one quarantining corruption applied."""
    row = draw(valid_rows())
    kind = draw(st.integers(min_value=0, max_value=6))
    if kind == 0:
        del row["asset_id"]
    elif kind == 1:
        row["asset_id"] = "   "
    elif kind == 2:
        row["model"] = ""
    elif kind == 3:
        del row["chain_of_custody"]
    elif kind == 4:
        row["chain_of_custody"] = []
    elif kind == 5:
        row["ecc_events"] = -draw(_counters) - 1
    else:
        row["thermal_history"] = [1.5]
    return row
