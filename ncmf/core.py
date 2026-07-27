# -*- coding: utf-8 -*-
"""
NCMF - Non-Compensatory Multimetric Framework for Technology Assessment
in Regulated Environments.

Reference implementation of the four indices described in the paper:
    TII    - Technology Implementability Index            (Eq. 1)
    V_reg  - Regulatory veto                              (Sec. 3.3)
    ISMI   - Integrated Systems Maturity Index            (Eq. 2, with AIF Eq. 3)
    CWRI   - Criticality-Weighted Risk Index              (Eq. 4)
    RG-MPI - Regulatory-Gated Mazziotta-Pareto Index      (Eq. 5)

Author: C. Gonzalez, P. Lamo, J. J. Rainer (Universidad Internacional de La Rioja)
License: MIT (code) / CC-BY-4.0 (data)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "System", "VetoChecklist", "tii", "aif", "operational_maturity",
    "dynamic_criticality", "ismi", "cwri", "rg_mpi", "assess",
]

# --------------------------------------------------------------------------- #
#  Stage 0 - Technology Implementability Index (Eq. 1)                          #
# --------------------------------------------------------------------------- #
def tii(M: float, A: float, V: float,
        w: Sequence[float] = (1 / 3, 1 / 3, 1 / 3), scale_max: int = 3) -> float:
    """Compensatory baseline index, Eq. (1). M, A, V are ordinal scores in [0, scale_max]."""
    w = np.asarray(w, float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("TII weights must sum to 1.")
    z = np.array([M, A, V], float) / scale_max
    if np.any((z < 0) | (z > 1)):
        raise ValueError(f"M, A, V must lie in [0, {scale_max}].")
    return float(100.0 * z @ w)


# --------------------------------------------------------------------------- #
#  Stage 1 - Regulatory veto (Sec. 3.3)                                         #
# --------------------------------------------------------------------------- #
@dataclass
class VetoChecklist:
    """Eight-item regulatory veto, each item anchored to auditable SFCR evidence.

    ``items``    : bool per veto item (True = evidence found).
    ``evidence`` : free-text SFCR section / page reference per item (audit trail).
    ``threshold``: minimum pass ratio; 1.0 = fully non-compensatory (paper default).
    """
    items: Sequence[bool]
    evidence: Sequence[str] = field(default_factory=list)
    threshold: float = 1.0

    LABELS = (
        "(i) IT governance and technology risk management",
        "(ii) Internal control system and compliance function",
        "(iii) Business continuity (BCM) and disaster recovery (ISO 22301)",
        "(iv) Management of critical ICT third parties",
        "(v) Cloud outsourcing management",
        "(vi) Inventory of essential ICT functions",
        "(vii) Security governance bodies (CISO / committee)",
        "(viii) Information integrity (SCIIF / UNE 19602)",
    )

    def __post_init__(self):
        if len(self.items) != 8:
            raise ValueError("The regulatory veto checklist has exactly 8 items.")

    @property
    def ratio(self) -> float:
        return float(np.mean(np.asarray(self.items, bool)))

    @property
    def passed(self) -> bool:
        return self.ratio >= self.threshold - 1e-12

    def failures(self) -> list[str]:
        return [lbl for lbl, ok in zip(self.LABELS, self.items) if not ok]


# --------------------------------------------------------------------------- #
#  Stage 2 - ISMI and CWRI                                                      #
# --------------------------------------------------------------------------- #
def operational_maturity(trl: np.ndarray, mrl: np.ndarray) -> np.ndarray:
    """OR_i = min(TRL_i, MRL_i) - component-level 'weakest link' logic."""
    return np.minimum(np.asarray(trl, float), np.asarray(mrl, float))


def aif(dsm: np.ndarray) -> np.ndarray:
    """Architectural Impact Factor, Eq. (3): AIF_i = 1 + sum_j w_ij * outdegree_j.

    ``dsm[i, j]`` in [0, 1] is the weight of the dependency of component *j* on
    component *i*; outdegree_j is the number of components that depend directly on *j*,
    i.e. the number of non-zero entries in row *j* of the DSM.
    """
    W = np.asarray(dsm, float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("DSM must be a square matrix.")
    outdeg = (W > 0).sum(axis=1)          # row j = components that depend on j
    return 1.0 + W @ outdeg


def dynamic_criticality(severity: np.ndarray, aif_vec: np.ndarray) -> np.ndarray:
    """C*_i = S_i * AIF_i  (FMEA severity anchored, DSM amplified)."""
    return np.asarray(severity, float) * np.asarray(aif_vec, float)


def ismi(op_maturity: np.ndarray, crit: np.ndarray) -> float:
    """Integrated Systems Maturity Index, Eq. (2)."""
    op_maturity, crit = np.asarray(op_maturity, float), np.asarray(crit, float)
    if crit.sum() <= 0:
        raise ValueError("Total dynamic criticality must be positive.")
    return float((op_maturity * crit).sum() / crit.sum())


def cwri(trl: np.ndarray, crit: np.ndarray, trl_target: int = 8,
         per_component: bool = False):
    """Criticality-Weighted Risk Index, Eq. (4).

        CWRI = sum_i  C*_i * ( max(0, TRL_target - TRL_i) )^2

    NOTE: the deficit is squared (non-linear penalty) and *multiplied* by the
    dynamic criticality. This is the formulation actually used to produce Table 3.
    """
    deficit = np.maximum(0.0, trl_target - np.asarray(trl, float)) ** 2
    per = np.asarray(crit, float) * deficit
    return per if per_component else float(per.sum())


# --------------------------------------------------------------------------- #
#  Stage 3 - Regulatory-Gated Mazziotta-Pareto Index (Eq. 5)                     #
# --------------------------------------------------------------------------- #
def rg_mpi(M: float, A: float, V: float,
           relevance: Sequence[float] = (1.0, 1.2, 1.5),
           v_threshold: float = 1.0) -> float:
    """RG-MPI, Eq. (5): mu - (lambda * sigma * cv), gated on the V dimension.

    ``M, A, V``    : normalised scores in [0, 1] (maturity, applicability, viability).
    ``relevance``  : penalty-relevance factors (p_M, p_A, p_V); lambda = p_k where k
                     is the dimension with the lowest score.
    ``v_threshold``: regulatory gate; returns NaN ('not viable') when V < threshold.
    """
    if V < v_threshold - 1e-12:
        return float("nan")                       # non-compensatory disqualification
    z = np.array([M, A, V], float)
    if np.any((z < 0) | (z > 1)):
        raise ValueError("M, A, V must be normalised to [0, 1].")
    mu, sigma = z.mean(), z.std()
    cv = sigma / mu if mu > 0 else 0.0
    lam = np.asarray(relevance, float)[int(np.argmin(z))]
    return float(mu - lam * sigma * cv)


# --------------------------------------------------------------------------- #
#  Full pipeline                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class System:
    """A candidate technology decomposed into n interdependent components."""
    name: str
    components: Sequence[str]
    trl: np.ndarray
    mrl: np.ndarray
    severity: np.ndarray            # FMEA severity, scale 1-10
    dsm: np.ndarray                 # n x n dependency-weight matrix
    veto: VetoChecklist
    applicability: float            # TII applicability score, scale 0-3
    trl_target: int = 8


def assess(sys: System, relevance=(1.0, 1.2, 1.5), v_threshold: float = 1.0) -> dict:
    """Run the complete NCMF pipeline and return an auditable result record."""
    result = {"system": sys.name,
              "veto_ratio": sys.veto.ratio,
              "veto_passed": sys.veto.passed,
              "veto_failures": sys.veto.failures()}

    if not sys.veto.passed:                      # Stage 1 stops the pipeline
        result.update(status="NOT VIABLE - regulatory veto",
                      ISMI=None, CWRI=None, RG_MPI=float("nan"))
        return result

    A_ = aif(sys.dsm)
    C_ = dynamic_criticality(sys.severity, A_)
    OR_ = operational_maturity(sys.trl, sys.mrl)
    result.update(
        status="VIABLE - proceeds to quantitative assessment",
        AIF=A_, C_dyn=C_, OR=OR_,
        ISMI=ismi(OR_, C_),
        CWRI=cwri(sys.trl, C_, sys.trl_target),
        CWRI_by_component=cwri(sys.trl, C_, sys.trl_target, per_component=True),
        RG_MPI=rg_mpi(M=float(OR_.min()) / 9.0,
                      A=sys.applicability / 3.0,
                      V=sys.veto.ratio,
                      relevance=relevance, v_threshold=v_threshold),
    )
    return result
