# Simulator test suite

Covers the `rechip01.simulator` module (spec §4): seeded fleet generation,
degradation physics, and the workload simulator. All results are simulated by
construction — `provenance=SIMULATED` is enforced at the batch level — and all
metrics are pipeline health checks, not reliability claims.

## Layout

| File | Scope |
|---|---|
| `conftest.py` | Shared scenario fixtures (4,000-unit built-in fleet, config overrides) |
| `test_generation.py` | Scenario tests: schema validity, label integrity, workload risk ordering, defect separation, label calibration |
| `test_determinism.py` | Bit-reproducibility under a fixed seed; label-only knobs leave features untouched; config validation |
| `test_knobs.py` | Monotonicity of the tunable rates (ECC/XID/defect/failure multipliers) with wide margins |
| `test_properties.py` | Hypothesis property tests: determinism, schema validity, and feature/label leakage over arbitrary valid configs |

## Go/no-go commands

The full gate is the repo standard (`ruff check .`, `ruff format --check .`,
`mypy`, `pytest`). The simulator-specific go/no-go scenarios are:

```bash
# Determinism + schema validity + label integrity at fleet scale (scenario suite)
pytest tests/test_simulator/test_generation.py tests/test_simulator/test_determinism.py -q

# Knob monotonicity: rates must move observed outcomes in the right direction
pytest tests/test_simulator/test_knobs.py -q

# Property tests over arbitrary valid configs (determinism, schema, leakage)
pytest tests/test_simulator/test_properties.py -q
```

A demonstration batch can be generated without pytest (spec §7 demo mode):

```bash
python -m rechip01.simulator --seed=1234 --n-units=4000 --out=/tmp/rechip-fleet
# writes scans.jsonl, labels.jsonl, and config.json under --out
```

## Interpretation guardrails

- Label calibration (realized failure fraction vs the generator's assigned
  `p_fail`) is a self-consistency check of the simulator, not model accuracy.
- Defect separation and knob monotonicity verify the ground-truth signal
  exists and moves with its knobs — the engine's evaluation harness (later
  PR) consumes these labels and must score against them, never train on them.
