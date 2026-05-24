"""Shared proportion statistics for paper figures."""

from __future__ import annotations

import math
from math import erfc, sqrt


def wilson_proportion_ci(
    k: int,
    n: int,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """
    Wilson score interval for proportion k/n.

    Returns (point_estimate, lower, upper) on the 0–1 scale.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_chi2(b: int, c: int, *, continuity_correction: bool = True) -> float:
    """
    McNemar chi-squared statistic from discordant counts b and c.

    Uses Yates continuity correction by default (matches paper Table 5.2).
    """
    n = b + c
    if n == 0:
        return float("nan")
    diff = abs(b - c)
    if continuity_correction and n > 0:
        diff = max(0.0, diff - 1.0)
    return diff**2 / n


def mcnemar_p_value(chi2: float) -> str:
    """Two-sided p-value string for chi-squared(1)."""
    if math.isnan(chi2):
        return "—"
    p = erfc(sqrt(chi2 / 2))
    if p < 0.001:
        return "p < .001"
    formatted = f"{p:.3f}"
    if formatted.startswith("0."):
        formatted = formatted[1:]
    return f"p = {formatted}"


def mcnemar_stat_line(title: str, b: int, c: int) -> str:
    """One-line McNemar summary for figure annotation."""
    chi2 = mcnemar_chi2(b, c)
    return f"{title}: χ²(1) = {chi2:.2f}, {mcnemar_p_value(chi2)}"
