# NCMF — Non-Compensatory Multimetric Framework for Technology Assessment in Regulated Environments

Reference implementation and full replication package for:

> González, C., Lamo, P., & Rainer, J. J. *A Non-Compensatory Multimetric Framework for
> Technology Assessment in Regulated Environments*. **ADCAIJ: Advances in Distributed
> Computing and Artificial Intelligence Journal**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What this repository contains

| Path | Content |
|---|---|
| `ncmf/core.py` | Reference implementation of TII (Eq. 1), the regulatory veto `V_reg` (Sec. 3.3), ISMI (Eq. 2), AIF (Eq. 3), CWRI (Eq. 4) and RG-MPI (Eq. 5) |
| `ncmf/baselines.py` | AHP/WSM, TOPSIS, PROMETHEE II (with optional outranking veto), ELECTRE-style veto, ungated MPI |
| `data/` | Component tables, Design Structure Matrices and the 8-item regulatory veto checklist with its SFCR evidence trail, for both pilots |
| `reproduce.py` | Single-command replication of **every** number in the paper: Tables 2–6, the sensitivity analysis, the comparative benchmark and the scalability measurements |

## Quick start

```bash
git clone https://github.com/<user>/ncmf-toolkit.git
cd ncmf-toolkit
pip install -r requirements.txt
python reproduce.py
```

Runtime: < 30 s on a standard laptop. All stochastic results are seeded
(`SEED = 20260729`) and therefore bit-for-bit reproducible.

## Minimal usage example

```python
import numpy as np
from ncmf import System, VetoChecklist, assess

sys = System(
    name="My platform",
    components=["Backend", "IoT", "Gateway", "Cloud", "Auth"],
    trl=np.array([6, 8, 7, 8, 9]),
    mrl=np.array([7, 8, 8, 9, 9]),
    severity=np.array([8, 7, 6, 10, 9]),      # FMEA severity, 1–10
    dsm=np.loadtxt("data/pilotB_dsm.csv", delimiter=",", skiprows=1, usecols=range(1, 6)),
    veto=VetoChecklist(items=[True] * 8),      # 8 auditable regulatory items
    applicability=2.0,                         # TII applicability score, 0–3
)
print(assess(sys))
```

If any veto item is `False`, `assess()` short-circuits and returns
`status = "NOT VIABLE - regulatory veto"` **without** computing an aggregate score.
This is the non-compensatory behaviour that defines the framework: no quantitative
strength can restore an alternative that fails an externally verifiable
regulatory requirement.

## Computational complexity

For *m* alternatives, *n* components and *q* veto items:

| Stage | Time | Space |
|---|---|---|
| TII (Eq. 1) | `O(d)`, d = 3 | `O(1)` |
| Regulatory veto (Sec. 3.3) | `O(q)`, short-circuits on first failure | `O(q)` |
| AIF from DSM (Eq. 3) | `O(n²)` | `O(n²)` |
| ISMI (Eq. 2), CWRI (Eq. 4) | `O(n)` | `O(n)` |
| RG-MPI (Eq. 5) | `O(d)` | `O(1)` |
| **Full pipeline** | **`O(m·n²)`** | **`O(n²)`** |

Measured: 0.05 ms at n = 5, 0.48 ms at n = 500, 79.6 ms at n = 5 000 (25 M DSM entries).
The DSM product dominates; a sparse DSM with `E` non-zero dependencies reduces this to
`O(n + E)`.

## Data provenance and licensing

The regulatory evidence used to instantiate the veto checklist comes from the
**Solvency and Financial Condition Report (SFCR)**, a document whose public disclosure
is *mandated* by Article 51 of Directive 2009/138/EC (Solvency II) and Articles 290–303
of Delegated Regulation (EU) 2015/35. Public availability is therefore a legal
obligation of the undertaking, and the report may be freely consulted, quoted and cited
for scientific purposes under Article 5(3)(d) of Directive 2001/29/EC.

`data/veto_checklist.csv` records only the **section and page references plus the
factual finding** for each veto item — it reproduces no substantial extract of the
report. Readers can verify every item against the public SFCR.

## License

- Code: MIT (see `LICENSE`)
- Data files produced by the authors: CC BY 4.0
