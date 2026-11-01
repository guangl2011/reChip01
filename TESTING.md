# Testing

Every deliverable ships with tests. This document maps each module to its
test suite and defines the go/no-go evaluation commands. All model metrics in
the MVP are computed against *simulated* ground truth and are labeled as such
everywhere they appear — they are pipeline health checks, not real-world
reliability results and not H1/H2/H3 outcomes.

## Module → suite map

| Module | Suite | Scope |
|---|---|---|
| `rechip01.health` | `tests/test_health.py` | Package imports; `/health` serves |
| `rechip01.schemas` | `tests/test_schemas.py` | `pytest` + `hypothesis`: JSON round-trips; provenance required and frozen on every contract; range/invariant rejections (ci brackets p_fail, unit-interval bounds, regret bounds) |
| `rechip01.repository` | `tests/test_repository.py` | Passport save/load parity; duplicate issuance rejected; latest-run lookup; path-safe record keys |
| `rechip01.simulator` | `tests/test_simulator/` (planned) | Schema validation on sampled outputs; seed reproducibility; ground-truth label integrity |
| `rechip01.scanner` | `tests/test_scanner/` (planned) | `pytest` + `hypothesis` property tests: valid/invalid/partial fixtures; provenance propagation |
| `rechip01.engine` | `tests/test_engine/` (planned) | Calibration report; AUROC vs simulator ground truth; OOD flagging |
| `rechip01.disposition` | `tests/test_disposition/` (planned) | Property tests on synthetic economies; hand-computed fixtures; regret-bound sanity |
| `rechip01.passport` (+ API) | `tests/test_api/` (planned) | `httpx`/`TestClient` integration; provenance immutability; OpenAPI schema check |
| Dashboard | UI smoke test (planned) | Batch overview + passport view render; simulated/measured badges visible |

## Commands

```bash
ruff check .            # lint (CI: lint job)
ruff format --check .   # formatting (CI: lint job)
mypy                    # strict typecheck (CI: typecheck job)
pytest                  # unit + property tests (CI: test job, py3.11 + py3.13)
```

## H1/H2/H3 comparison plan

The evaluation harness (built alongside the engine) must be able to reproduce
the proposal's hypothesis comparisons from day one, on identical held-out
simulator data:

- **H1 scaffold** — engine output vs a DCGM-threshold-style baseline and a
  simple error-count heuristic, run as one command, emitting a report.
- **H2 / H3** — thresholds and comparison definitions follow the proposal
  (§3); all numeric thresholds are provisional until the Phase I pilot-dataset
  power analysis.

Calibration thresholds are configuration values: set conservatively at first
and tuned only against the simulator's held-out ground truth, never against
training performance.
