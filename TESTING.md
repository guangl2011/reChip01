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
| `rechip01.repository` | `tests/test_repository.py` | Passport and scan-record save/load parity; duplicate issuance rejected; latest-run lookup; path-safe record keys |
| `rechip01.simulator` | `tests/test_simulator/` | Schema validation on sampled outputs; seed reproducibility; ground-truth label integrity; knob monotonicity (see `tests/test_simulator/README.md` for the go/no-go commands) |
| `rechip01.scanner` | `tests/test_scanner/` | `pytest` + `hypothesis`: valid/invalid/partial fixtures; header normalization and string coercion; provenance stamping and conflict quarantine; machine-readable rejection reasons; feed-vs-export parity; repository round-trips; CLI end-to-end |
| `rechip01.engine` | `tests/test_engine/` | `pytest` + `hypothesis`: MLE risk ordering vs generator truth; interval bracketing/ordering/widening; OOD flag never scores silently (valid intervals, widening supersets); missing-input flagging; provenance propagation; seed determinism; harness rails — AUROC CI excludes 0.5 with a wide-margin point floor, ECE, coverage, in-distribution OOD rate (see `tests/test_engine/test_evaluation.py`) |
| `rechip01.evaluation` | `tests/test_baseline_comparison.py`, `tests/test_baseline_properties.py` | `pytest` + `hypothesis`: end-to-end engine vs DCGM-threshold-grading vs error-count heuristic on the identical held-out batch; metric invariants (AUROC bounds/reversal symmetry/monotone invariance, ECE bounds); holdout isolation — baseline scores depend only on held-out scans and never shift with the training fleet; CLI report + JSON generation (see below) |
| `rechip01.disposition` | `tests/test_disposition/` | `pytest` + `hypothesis`: price-range/economy/config contract validation; hand-computed fixture valuations on a degenerate economy; Monte Carlo argmax vs a brute-force point-probability grid (every economic path wins somewhere); review routing — OOD flag and headline CI width route to `manual_review` (risk level alone never does), review cost priced into the decision; determinism under fixed seed; MC mean convergence to the point expectation; monotonicity — reuse ENR falls as fitness degrades, value spread widens with probability spread; evidence trace cites estimate, economy, sensitivity, and regret lines on every decision; H3-shaped policy comparison — pass/fail-resale vs optimizer vs ground-truth oracle on the identical held-out fleet with paired bootstrap differences and CLI report/JSON (see below) |
| `rechip01.passport` + API | `tests/test_api/` | `httpx`/`TestClient` integration: `/scan`, `/batch`, `/assess`, `/passport/{asset_id}`, `/health` workflows; builder binds scan, estimate, and decision with a SHA-256 lineage hash and model version; provenance immutable after issuance; OpenAPI schema check; Hypothesis property tests on the request/response contracts |
| `rechip01.dashboard` | `tests/test_dashboard/` | State derivation (pending/partial/ready/review); batch overview renders provenance and state badges plus uncertainty; passport page renders estimate, disposition, and evidence trace; rendered 404 for missing passports; provenance badges visible on every screen |

## Commands

All commands are runnable from a clean checkout after the one-time install:

```bash
python -m pip install -e ".[dev]"
```

```bash
ruff check .            # lint (CI: lint job)
ruff format --check .   # formatting (CI: lint job)
mypy                    # strict typecheck (CI: typecheck job)
pytest                  # unit + property tests (CI: test job, py3.11 + py3.13)

# Engine go/no-go evaluation — trains on a simulated fleet, scores a held-out
# batch with injected latent defects, and writes calibration-report.md,
# estimates.jsonl, and config.json (CI: evaluation-report job, artifact).
python -m rechip01.engine --out=calibration-report

# Baseline comparison (H1 scaffold) — the engine vs a DCGM-threshold-style
# grading rule and an error-count heuristic on the identical held-out batch;
# writes baseline-comparison.md and baseline-metrics.json (CI:
# evaluation-report job, artifact). All output is a simulation-based pipeline
# check, not an H1/H2/H3 outcome.
python -m rechip01.evaluation --out=baseline-comparison

# Disposition comparison (H3-shaped) — the risk-adjusted optimizer vs a
# simple pass/fail resale policy and a ground-truth oracle on the identical
# held-out fleet under a pre-registered synthetic economy; writes
# disposition-comparison.md and disposition-metrics.json. Simulation-based
# pipeline check only — never an H3 outcome.
python -m rechip01.disposition --out=disposition-comparison
```

## H1/H2/H3 comparison plan

The evaluation harness (built alongside the engine) must be able to reproduce
the proposal's hypothesis comparisons from day one, on identical held-out
simulator data:

- **H1 scaffold** — engine output vs a DCGM-threshold-style baseline and a
  simple error-count heuristic, run as one command, emitting a report.
- **H3-shaped disposition comparison** — the risk-adjusted optimizer vs a
  simple pass/fail resale policy (gate on modeled risk, recycle below it)
  and a ground-truth oracle, on identical held-out data under one
  pre-registered synthetic economy, with paired bootstrap differences.
- **H2 / H3** — thresholds and comparison definitions follow the proposal
  (§3); all numeric thresholds are provisional until the Phase I pilot-dataset
  power analysis.

Calibration thresholds are configuration values: set conservatively at first
and tuned only against the simulator's held-out ground truth, never against
training performance.

## Spec §5 acceptance criteria → named tests and commands

Each acceptance criterion in the product spec's deliverables table maps to a
named test or command a reviewer can run (all under `pytest`, or one of the
commands above):

| Spec deliverable | Named test(s) | Runnable check |
|---|---|---|
| Fleet simulator | `tests/test_simulator/test_generation.py` (schema on sampled outputs), `tests/test_simulator/test_determinism.py` (seeded runs bit-reproducible), `tests/test_simulator/test_knobs.py` (≥2 models, ≥3 workloads, tunable defect/failure rates), `tests/test_simulator/test_properties.py` (ground-truth label integrity) | `pytest tests/test_simulator/` |
| Scanner | `tests/test_scanner/test_validation.py`, `test_normalization.py`, `test_provenance.py`, `test_quarantine.py`, `test_cli.py`, `test_properties.py` (valid/invalid/partial fixtures, machine-readable reasons, provenance stamping, feed-vs-export parity) | `pytest tests/test_scanner/` |
| Reliability engine | `tests/test_engine/test_evaluation.py` (AUROC CI excludes 0.5, ECE, coverage, OOD rate), `test_ood.py` (OOD never scored silently), `test_model.py`, `test_metrics.py`, `test_properties.py` | `pytest tests/test_engine/` and `python -m rechip01.engine --out=calibration-report` |
| Baseline comparison (H1 scaffold) | `tests/test_baseline_comparison.py`, `tests/test_baseline_properties.py` (identical held-out data; report generation asserted) | `python -m rechip01.evaluation --out=baseline-comparison` |
| Disposition optimizer | `tests/test_disposition/test_optimizer.py` (hand-computed fixtures, review routing), `test_properties.py` (regret-bound sanity, monotonicity), `test_config.py` | `pytest tests/test_disposition/` and `python -m rechip01.disposition --out=disposition-comparison` |
| Asset passport + API | `tests/test_api/test_builder.py` (evidence trace, model version, lineage hash), `test_passport.py` (provenance immutability), `test_openapi.py` (OpenAPI schema), `test_scan.py` / `test_assess.py` / `test_batch.py` | `pytest tests/test_api/` |
| Dashboard | `tests/test_dashboard/test_state.py`, `test_batch.py`, `test_passport.py` (provenance badges on every screen) | `pytest tests/test_dashboard/` |
| Testing plan document | This document; CI (`.github/workflows/ci.yml`) runs exactly the mapped suites: lint, typecheck, test (py3.11 + py3.13), and the three evaluation commands as the `evaluation-report` job with artifact uploads | `ruff check . && ruff format --check . && mypy && pytest` |

Every metric these commands produce is computed against simulated ground truth
and is a simulation-based pipeline check — see the honesty-rails note above.
