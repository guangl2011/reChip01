# reChip01

Simulation-first reliability assessment and disposition decisions for retired
AI servers and GPUs. reChip01 turns an ITAD's existing test evidence and
telemetry into forward-looking, workload-conditioned reliability estimates —
then prices the disposition decision (reuse / repair / harvest / recycle) with
explicit uncertainty and an auditable asset passport per unit.

> **Status:** pre-alpha scaffold. The pipeline is being built simulation-first:
> a physics-grounded fleet simulator produces labeled synthetic assets, and
> real partner data replaces simulated inputs through the same schema without
> downstream changes. Every field and report carries provenance
> (`simulated` / `measured`); the system never issues accredited
> third-party certification.

## Pipeline

```
scanner ──▶ engine ──▶ disposition ──▶ passport
   ▲                                      │
   └──────── simulator (demo mode)        ▼
                              API (FastAPI) + dashboard
```

| Module | Owns |
|---|---|
| `rechip01.scanner` | Inventory/telemetry ingest, validation, provenance stamping |
| `rechip01.simulator` | Physics-grounded synthetic fleets with ground-truth labels |
| `rechip01.engine` | Uncertainty-aware, workload-conditioned reliability model |
| `rechip01.disposition` | Risk-adjusted net-recovery optimizer with regret bounds |
| `rechip01.passport` | Auditable asset passports served over FastAPI |

## Development

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"

ruff check .        # lint
ruff format --check .
mypy                # strict typecheck
pytest              # unit + property tests

# Go/no-go evaluation commands (simulation-based pipeline checks, not
# real-world reliability results or H1/H2/H3 outcomes):
python -m rechip01.engine --out=calibration-report
python -m rechip01.evaluation --out=baseline-comparison
python -m rechip01.disposition --out=disposition-comparison
```

CI runs ruff, mypy (strict), and pytest on Python 3.11 and 3.13 for every
push and pull request targeting `main`, plus the three evaluation commands
above as an artifact-producing job. [TESTING.md](TESTING.md) maps every
module to its test suite, every spec acceptance criterion to a named test or
command, and states the H1/H2/H3 comparison plan.

## License

MIT — see [LICENSE](LICENSE).
