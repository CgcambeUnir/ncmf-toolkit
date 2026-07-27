# -*- coding: utf-8 -*-
"""
Full replication of every quantitative result reported in:

  Gonzalez, C., Lamo, P., & Rainer, J. J. "A Non-Compensatory Multimetric Framework
  for Technology Assessment in Regulated Environments." ADCAIJ.

Run:  python reproduce.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ncmf.core import System, VetoChecklist, assess, aif, dynamic_criticality, \
    operational_maturity, ismi, cwri, rg_mpi
from ncmf import baselines as bl

DATA = Path(__file__).parent / "data"
SEED = 20260729
rng = np.random.default_rng(SEED)
TRL_TARGET = 8
BENEFIT = np.array([True, False, True, True])      # ISMI+, CWRI-, compliance+, applicability+
W0 = np.array([0.30, 0.25, 0.25, 0.20])


def load(tag: str, applicability: float) -> System:
    comp = pd.read_csv(DATA / f"pilot{tag}_components.csv")
    dsm = pd.read_csv(DATA / f"pilot{tag}_dsm.csv", index_col=0)
    chk = pd.read_csv(DATA / "veto_checklist.csv")
    return System(
        name=f"Pilot {tag}",
        components=comp.component_name.tolist(),
        trl=comp.TRL.to_numpy(float), mrl=comp.MRL.to_numpy(float),
        severity=comp.severity_FMEA.to_numpy(float), dsm=dsm.to_numpy(float),
        veto=VetoChecklist(items=chk[f"pilot{tag}_pass"].astype(bool).tolist(),
                           evidence=chk[f"pilot{tag}_evidence"].tolist()),
        applicability=applicability, trl_target=TRL_TARGET,
    )


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1. pipeline
pilotA, pilotB = load("A", applicability=3.0), load("B", applicability=2.0)

banner("1. NCMF PIPELINE  (reproduces Table 2 and Table 3)")
for s in (pilotA, pilotB):
    r = assess(s)
    print(f"\n--- {r['system']} ---")
    print(f"  Regulatory veto: {r['veto_ratio']*8:.0f}/8 items -> {r['status']}")
    for f in r["veto_failures"]:
        print(f"     FAILED: {f}")
    if r["ISMI"] is not None:
        print(f"  {'component':34s} {'OR':>4s} {'AIF':>6s} {'C*':>7s} {'CWRI_i':>8s}")
        for n_, o, a_, c_, k_ in zip(s.components, r["OR"], r["AIF"],
                                     r["C_dyn"], r["CWRI_by_component"]):
            print(f"  {n_:34s} {o:4.0f} {a_:6.2f} {c_:7.1f} {k_:8.1f}")
        print(f"  {'TOTAL':34s} {'':4s} {'':6s} {r['C_dyn'].sum():7.1f} {r['CWRI']:8.1f}")
        print(f"  ISMI = {r['ISMI']:.2f}   CWRI = {r['CWRI']:.1f}   RG-MPI = {r['RG_MPI']:.4f}")

# retrospective technical assessment of the vetoed Pilot A (needed for benchmarking)
A_aif = aif(pilotA.dsm)
A_C = dynamic_criticality(pilotA.severity, A_aif)
A_OR = operational_maturity(pilotA.trl, pilotA.mrl)
A_ismi, A_cwri = ismi(A_OR, A_C), cwri(pilotA.trl, A_C, TRL_TARGET)
B_aif = aif(pilotB.dsm)
B_C = dynamic_criticality(pilotB.severity, B_aif)
B_OR = operational_maturity(pilotB.trl, pilotB.mrl)
B_ismi, B_cwri = ismi(B_OR, B_C), cwri(pilotB.trl, B_C, TRL_TARGET)
B_per = cwri(pilotB.trl, B_C, TRL_TARGET, per_component=True)

print(f"\n  [retrospective] Pilot A ISMI = {A_ismi:.2f}, CWRI = {A_cwri:.1f}"
      f"  (computed post hoc; the framework had already vetoed it)")

# ---------------------------------------------------------------- 2. benchmark
banner("2. COMPARATIVE BENCHMARK AGAINST ESTABLISHED MCDA METHODS (Table 5)")
names = ["Pilot A (AI Platform)", "Pilot B (Connected Mobility)"]
X = np.array([[A_ismi, A_cwri, pilotA.veto.ratio * 100, pilotA.applicability / 3 * 100],
              [B_ismi, B_cwri, pilotB.veto.ratio * 100, pilotB.applicability / 3 * 100]])
print("Decision matrix [ISMI, CWRI, compliance %, applicability %]:")
print(pd.DataFrame(X, index=names,
                   columns=["ISMI", "CWRI", "Compliance", "Applicability"]).round(2), "\n")

Z = bl.normalise(X, BENEFIT)
methods = {
    "AHP / WSM":            bl.wsm(X, W0, BENEFIT),
    "TOPSIS":               bl.topsis(X, W0, BENEFIT),
    "PROMETHEE II":         bl.promethee_ii(X, W0, BENEFIT),
    "MPI (ungated)":        bl.mpi(Z),
    "ELECTRE (veto set)":   bl.electre_veto(X, W0, BENEFIT, [0.0, 0.0, 0.999, 0.0])[0],
    "NCMF (RG-MPI)":        np.array([rg_mpi(A_OR.min() / 9, pilotA.applicability / 3,
                                             pilotA.veto.ratio),
                                      rg_mpi(B_OR.min() / 9, pilotB.applicability / 3,
                                             pilotB.veto.ratio)]),
}
print(f"{'Method':22s} {'Pilot A':>10s} {'Pilot B':>10s}  Ranked 1st"
      f"                     Matches reality?")
for k, v in methods.items():
    top = names[int(np.nanargmax(v))] if not np.all(np.isnan(v)) else "none"
    ok = "YES" if "Pilot B" in top else "NO  <-- rank reversal"
    a_ = "vetoed" if np.isnan(v[0]) else f"{v[0]:.4f}"
    b_ = "vetoed" if np.isnan(v[1]) else f"{v[1]:.4f}"
    print(f"{k:22s} {a_:>10s} {b_:>10s}  {top:32s} {ok}")
print("\nGround truth: Pilot A was REJECTED by the organisation (data-governance and "
      "third-party risk);\n              Pilot B went live (with incidents).")

# ---------------------------------------------------------------- 3. Monte Carlo
banner("3. MONTE-CARLO SENSITIVITY ANALYSIS, K = 10 000 (Table 4)")
K = 10_000
hits = {k: 0 for k in ["AHP / WSM", "TOPSIS", "PROMETHEE II", "MPI (ungated)"]}
ncmf_ok = 0
for _ in range(K):
    w = rng.dirichlet(np.ones(4) * 4)                     # criterion weights
    t = int(rng.choice([7, 8, 9]))                        # TRL target
    pV, pA, pM = rng.uniform(1.0, 2.0), rng.uniform(1.0, 1.5), rng.uniform(0.8, 1.2)
    Xs = np.array([[ismi(A_OR, A_C), cwri(pilotA.trl, A_C, t),
                    pilotA.veto.ratio * 100, pilotA.applicability / 3 * 100],
                   [ismi(B_OR, B_C), cwri(pilotB.trl, B_C, t),
                    pilotB.veto.ratio * 100, pilotB.applicability / 3 * 100]])
    if bl.wsm(Xs, w, BENEFIT)[0] > bl.wsm(Xs, w, BENEFIT)[1]:               hits["AHP / WSM"] += 1
    if bl.topsis(Xs, w, BENEFIT)[0] > bl.topsis(Xs, w, BENEFIT)[1]:         hits["TOPSIS"] += 1
    if bl.promethee_ii(Xs, w, BENEFIT)[0] > bl.promethee_ii(Xs, w, BENEFIT)[1]:
        hits["PROMETHEE II"] += 1
    Zs = bl.normalise(Xs, BENEFIT)
    if bl.mpi(Zs)[0] > bl.mpi(Zs)[1]:                                       hits["MPI (ungated)"] += 1
    rel = (pM, pA, pV)
    sA = rg_mpi(A_OR.min() / 9, pilotA.applicability / 3, pilotA.veto.ratio, rel)
    sB = rg_mpi(B_OR.min() / 9, pilotB.applicability / 3, pilotB.veto.ratio, rel)
    if np.isnan(sA) and not np.isnan(sB):
        ncmf_ok += 1
for k, v in hits.items():
    print(f"  {k:22s}: ranks the REJECTED Pilot A first in {100*v/K:6.2f}% of draws")
print(f"  {'NCMF (RG-MPI)':22s}: correct classification in {100*ncmf_ok/K:6.2f}% of draws")

# ---------------------------------------------------------------- 4. OAT + veto
banner("4. ONE-AT-A-TIME SENSITIVITY, PILOT B (Table 6)")
base_i, base_c = B_ismi, B_cwri
for k in ("TRL", "MRL", "severity", "AIF"):
    for d in (-1, +1):
        trl, mrl, sev, A_ = (pilotB.trl.copy(), pilotB.mrl.copy(),
                             pilotB.severity.copy(), B_aif.copy())
        if k == "TRL":       trl = np.clip(trl + d, 1, 9)
        elif k == "MRL":     mrl = np.clip(mrl + d, 1, 9)
        elif k == "severity": sev = np.clip(sev + d, 1, 10)
        else:                A_ = np.clip(A_ + 0.2 * d, 1, 3)
        C_ = sev * A_
        i_, c_ = ismi(operational_maturity(trl, mrl), C_), cwri(trl, C_, TRL_TARGET)
        print(f"  {k:9s} {d:+d}:  ISMI = {i_:5.2f} ({100*(i_-base_i)/base_i:+6.2f}%)"
              f"   CWRI = {c_:6.1f} ({100*(c_-base_c)/base_c:+7.2f}%)")

rho, c1_top = [], 0
for _ in range(2000):
    trl = np.clip(pilotB.trl + rng.integers(-1, 2, 5), 1, 9)
    sev = np.clip(pilotB.severity + rng.integers(-1, 2, 5), 1, 10)
    A_ = np.clip(B_aif + rng.normal(0, .15, 5), 1, 3)
    per = cwri(trl, sev * A_, TRL_TARGET, per_component=True)
    r = spearmanr(B_per, per).statistic
    if not np.isnan(r):
        rho.append(r)
    if np.argmax(per) == 0:
        c1_top += 1
rho = np.array(rho)
print(f"\n  Component-risk ranking under joint noise (2 000 replications):")
print(f"     Spearman rho vs baseline: mean = {rho.mean():.3f}, 5th pct = {np.percentile(rho,5):.3f}")
print(f"     C1 remains the dominant risk driver in {100*c1_top/2000:.1f}% of replications")
print(f"     C1 share of baseline CWRI = {100*B_per[0]/base_c:.1f}%")

banner("5. VETO-THRESHOLD SENSITIVITY")
for thr in (1.0, 7 / 8, 6 / 8, 5 / 8):
    a = "PASS" if pilotA.veto.ratio >= thr - 1e-9 else "VETOED"
    b = "PASS" if pilotB.veto.ratio >= thr - 1e-9 else "VETOED"
    print(f"  V_threshold = {thr:.3f} ({thr*8:.0f}/8 items): Pilot A {a:6s} | Pilot B {b}")
print("  NOTE: items (iv) and (v) implement DORA Arts. 28-30 (mandatory). Relaxing the")
print("        threshold below 8/8 has no legal basis in the DORA/Solvency II setting.")

# ---------------------------------------------------------------- 6. complexity
banner("6. EMPIRICAL SCALABILITY OF THE PIPELINE  -  O(m . n^2)")
print(f"  {'n components':>14s} {'DSM entries':>14s} {'runtime':>12s}")
for n in (5, 50, 500, 2000, 5000):
    trl = rng.integers(3, 10, n); mrl = rng.integers(3, 10, n)
    sev = rng.integers(1, 11, n).astype(float)
    W = rng.random((n, n)) * (rng.random((n, n)) < 0.1)
    np.fill_diagonal(W, 0.0)
    t0 = time.perf_counter()
    C_ = dynamic_criticality(sev, aif(W))
    ismi(operational_maturity(trl, mrl), C_); cwri(trl, C_, TRL_TARGET)
    dt = (time.perf_counter() - t0) * 1000
    print(f"  {n:>14,d} {n*n:>14,d} {dt:>10.2f} ms")

print("\nDone. All figures above are deterministic under SEED =", SEED)
