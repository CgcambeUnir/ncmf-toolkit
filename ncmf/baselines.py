# -*- coding: utf-8 -*-
"""Established MCDA baselines used for the comparative validation (Sec. 4.5)."""
from __future__ import annotations

import itertools
import numpy as np

__all__ = ["normalise", "wsm", "topsis", "promethee_ii", "electre_veto", "mpi"]

# Global anchors keep min-max normalisation meaningful with few alternatives.
ANCHOR_LO = np.array([0.0, 0.0, 0.0, 0.0])
ANCHOR_HI = np.array([9.0, 120.0, 100.0, 100.0])   # ISMI, CWRI, compliance %, applicability %


def normalise(X, benefit, lo=ANCHOR_LO, hi=ANCHOR_HI):
    Z = (np.asarray(X, float) - lo) / (hi - lo)
    return np.where(benefit, Z, 1.0 - Z)


def wsm(X, w, benefit):
    """Weighted sum (aggregation stage of AHP with given priority vector)."""
    return normalise(X, benefit) @ np.asarray(w, float)


def topsis(X, w, benefit):
    X = np.asarray(X, float)
    V = X / np.sqrt((X ** 2).sum(0)) * np.asarray(w, float)
    ideal = np.where(benefit, V.max(0), V.min(0))
    anti = np.where(benefit, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dn = np.sqrt(((V - anti) ** 2).sum(1))
    return dn / (dp + dn)


def promethee_ii(X, w, benefit, q=None, p=None, veto=None):
    """PROMETHEE II net outranking flow, V-shape preference with indifference.

    ``veto`` (optional, per criterion) reproduces the classical outranking veto:
    if the deviation against alternative *a* on any criterion exceeds the veto
    threshold, *a* cannot outrank *b*.
    """
    X = np.asarray(X, float)
    m, n = X.shape
    q = np.zeros(n) if q is None else np.asarray(q, float)
    p = (np.abs(X).max(0) * 0.5 + 1e-9) if p is None else np.asarray(p, float)
    Pi = np.zeros((m, m))
    for a, b in itertools.permutations(range(m), 2):
        s, blocked = 0.0, False
        for j in range(n):
            d = X[a, j] - X[b, j]
            if not benefit[j]:
                d = -d
            if veto is not None and -d >= veto[j]:
                blocked = True
            s += w[j] * (0.0 if d <= q[j] else 1.0 if d >= p[j] else (d - q[j]) / (p[j] - q[j]))
        Pi[a, b] = 0.0 if blocked else s
    return Pi.sum(1) / (m - 1) - Pi.sum(0) / (m - 1)


def electre_veto(X, w, benefit, veto_thr):
    """Weighted concordance score with a hard per-criterion veto (ELECTRE-style)."""
    Z = normalise(X, benefit)
    score = Z @ np.asarray(w, float)
    killed = (Z < np.asarray(veto_thr, float)).any(axis=1)
    return np.where(killed, np.nan, score), killed


def mpi(Z, lam=1.0):
    """Standard (ungated) Mazziotta-Pareto Index on normalised scores."""
    Z = np.asarray(Z, float)
    mu, sd = Z.mean(1), Z.std(1)
    cv = np.divide(sd, mu, out=np.zeros_like(mu), where=mu > 0)
    return mu - lam * sd * cv
