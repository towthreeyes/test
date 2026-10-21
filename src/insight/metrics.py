"""Core statistics for the ledger-lag -> missing-part analysis.

Conventions
-----------
* ``clean`` frame: exact duplicate rows removed, rows with missing ``assembled``
  dropped.  This is the frame every reported rate uses.
* ``delay_min``: age of the ledger snapshot when the dispatch was requested.
* A "hit" is an intercepted dispatch whose outcome was ``assembled == 0``.

All money/time numbers are minutes of station downtime:
``BASELINE_COST_MIN = 75`` per stoppage, ``INTERCEPT_COST_MIN = 12`` per
intercepted dispatch.  Both are assumptions; they are parameters here so the
sensitivity can be shown in the README instead of being hidden in the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASELINE_COST_MIN = 75.0
INTERCEPT_COST_MIN = 12.0


# ---------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates and rows without an assembled outcome.

    Duplicates are detected on every column except the derived delay, so two
    real dispatches that differ in any raw field are kept.  The 49 NaNs are
    dropped: an unknown outcome is neither a stoppage nor a success and would
    bias the rate if counted either way.
    """
    cols = [c for c in df.columns if c != "delay_min"]
    out = df.drop_duplicates(subset=cols, keep="first").copy()
    out = out.dropna(subset=["assembled"]).copy()
    out["assembled"] = out["assembled"].astype(int)
    return out.reset_index(drop=True)


def clean_summary(df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    raw = len(df)
    dup = raw - len(df.drop_duplicates(subset=[c for c in df.columns if c != "delay_min"]))
    raw_eff = df.dropna(subset=["assembled"])
    return {
        "raw_rows": raw,
        "duplicate_rows_removed": int(dup),
        "missing_outcome_rows": int(df["assembled"].isna().sum()),
        "clean_rows": len(clean_df),
        "clean_stoppages": int((clean_df["assembled"] == 0).sum()),
        "clean_assembled": int((clean_df["assembled"] == 1).sum()),
        "clean_missing_rate": float((clean_df["assembled"] == 0).mean()),
        "raw_missing_rate_incl_missing_as_denominator": float(
            (df["assembled"] == 0).sum() / raw
        ),
        "raw_missing_rate_valid_denominator": float(
            (raw_eff["assembled"] == 0).mean()
        ),
        "delay_mean": float(clean_df["delay_min"].mean()),
        "delay_p50": float(clean_df["delay_min"].median()),
        "delay_p95": float(clean_df["delay_min"].quantile(0.95)),
        "delay_max": float(clean_df["delay_min"].max()),
    }


# ------------------------------------------------------------- intervals
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no scipy needed)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def chi2_p_value(table: np.ndarray) -> tuple[float, float, int]:
    """Pearson chi-square test on a 2xK contingency table.

    Returns (chi2, p, dof).  The p-value uses the normal tail approximation,
    accurate to ~1e-7 for dof > 3 and adequate at dof = 1..2 for screening.
    """
    k = table.astype(float)
    total = k.sum()
    if total <= 0 or k.shape[0] < 2 or k.shape[1] < 2:
        return (0.0, 1.0, 0)
    expected = k.sum(axis=1, keepdims=True) * k.sum(axis=0, keepdims=True) / total
    mask = expected > 0
    chi2 = float(((k[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    dof = (k.shape[0] - 1) * (k.shape[1] - 1)
    if dof == 1:
        p = math.erfc(math.sqrt(chi2 / 2.0))
    elif dof == 2:
        p = math.exp(-chi2 / 2.0)
    else:
        # Wilson-Hilferty: chi2 ~ dof * (1 - 2/(9dof) + z*sqrt(2/(9dof)))^3
        z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
        p = 0.5 * math.erfc(z / math.sqrt(2))
    return (chi2, p, dof)


def cramers_v(table: np.ndarray) -> float:
    k = table.astype(float)
    n = k.sum()
    if n == 0:
        return 0.0
    expected = k.sum(axis=1, keepdims=True) * k.sum(axis=0, keepdims=True) / n
    mask = expected > 0
    chi2 = float(((k[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    return math.sqrt(max(0.0, chi2 / (n * (min(k.shape) - 1))))


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """OR for [[a,b],[c,d]] with Haldane-Anscombe correction on zeros."""
    a, b, c, d = (x + 0.5 if x == 0 else float(x) for x in (a, b, c, d))
    return (a * d) / (b * c)


# ---------------------------------------------------------------- buckets
def delay_bucket_curve(
    df: pd.DataFrame,
    edges: list[float] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Missing-part rate by ledger-lag bucket, with Wilson 95% CIs."""
    edges = edges or [0, 5, 10, 15, 20, 30, 45, 60, 71]
    labels = labels or [f"{a:g}-{b:g}" for a, b in zip(edges[:-1], edges[1:])]
    cut = pd.cut(df["delay_min"], bins=edges, labels=labels, right=False)
    g = df.groupby(cut, observed=True)["assembled"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "stoppages": g.apply(lambda s: int((s == 0).sum())),
        }
    )
    out["missing_rate"] = out["stoppages"] / out["n"]
    ci = [wilson_interval(int(s), int(n)) for s, n in zip(out["stoppages"], out["n"])]
    out["ci_low"], out["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return out.reset_index().rename(columns={"delay_min": "delay_bucket"})


def two_group_test(df: pd.DataFrame, threshold: float) -> dict:
    """Compare missing rate below vs at/above the ledger-lag threshold."""
    hi = df[df["delay_min"] >= threshold]
    lo = df[df["delay_min"] < threshold]
    a = int((hi["assembled"] == 0).sum())
    b = len(hi) - a
    c = int((lo["assembled"] == 0).sum())
    d = len(lo) - c
    table = np.array([[a, b], [c, d]])
    chi2, p, dof = chi2_p_value(table)
    rate_hi = a / len(hi) if len(hi) else float("nan")
    rate_lo = c / len(lo) if len(lo) else float("nan")
    return {
        "threshold": threshold,
        "n_high": len(hi),
        "n_low": len(lo),
        "stoppages_high": a,
        "stoppages_low": c,
        "rate_high": rate_hi,
        "rate_low": rate_lo,
        "rate_ratio": (rate_hi / rate_lo) if rate_lo else float("nan"),
        "odds_ratio": odds_ratio(a, b, c, d),
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": cramers_v(table),
        "ci_high": wilson_interval(a, len(hi)),
        "ci_low": wilson_interval(c, len(lo)),
    }


# --------------------------------------------------------------- threshold
@dataclass
class ThresholdResult:
    threshold: float
    intercepted: int
    true_hits: int
    false_alarms: int
    hit_rate: float
    precision_ci: tuple[float, float]
    recall: float
    net_savings_min: float
    savings_vs_baseline: float
    cost_of_intercepts_min: float


def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
    baseline_cost: float = BASELINE_COST_MIN,
    intercept_cost: float = INTERCEPT_COST_MIN,
) -> ThresholdResult:
    """Apply "intercept if delay_min >= threshold" and score it."""
    total_stops = int((df["assembled"] == 0).sum())
    flag = df["delay_min"] >= threshold
    n_int = int(flag.sum())
    hits = int(((df["assembled"] == 0) & flag).sum())
    fa = n_int - hits
    net = hits * baseline_cost - n_int * intercept_cost
    baseline = total_stops * baseline_cost
    return ThresholdResult(
        threshold=float(threshold),
        intercepted=n_int,
        true_hits=hits,
        false_alarms=fa,
        hit_rate=(hits / n_int) if n_int else 0.0,
        precision_ci=wilson_interval(hits, n_int),
        recall=(hits / total_stops) if total_stops else 0.0,
        net_savings_min=net,
        savings_vs_baseline=(net / baseline) if baseline else 0.0,
        cost_of_intercepts_min=n_int * intercept_cost,
    )


def threshold_sweep(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
    baseline_cost: float = BASELINE_COST_MIN,
    intercept_cost: float = INTERCEPT_COST_MIN,
) -> pd.DataFrame:
    thresholds = thresholds or list(range(0, 61))
    rows = []
    for t in thresholds:
        r = evaluate_threshold(df, t, baseline_cost, intercept_cost)
        rows.append(
            {
                "threshold": r.threshold,
                "intercepted": r.intercepted,
                "true_hits": r.true_hits,
                "false_alarms": r.false_alarms,
                "hit_rate": r.hit_rate,
                "hit_rate_ci_low": r.precision_ci[0],
                "hit_rate_ci_high": r.precision_ci[1],
                "recall": r.recall,
                "net_savings_min": r.net_savings_min,
                "savings_vs_baseline": r.savings_vs_baseline,
                "cost_of_intercepts_min": r.cost_of_intercepts_min,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(sweep: pd.DataFrame) -> dict:
    best = sweep.loc[sweep["net_savings_min"].idxmax()]
    # plateau: every threshold within 1% of the max net saving
    near = sweep[sweep["net_savings_min"] >= 0.99 * best["net_savings_min"]]
    return {
        "threshold": float(best["threshold"]),
        "intercepted": int(best["intercepted"]),
        "true_hits": int(best["true_hits"]),
        "hit_rate": float(best["hit_rate"]),
        "false_alarms": int(best["false_alarms"]),
        "net_savings_min": float(best["net_savings_min"]),
        "savings_vs_baseline": float(best["savings_vs_baseline"]),
        "plateau_min": float(near["threshold"].min()),
        "plateau_max": float(near["threshold"].max()),
        "breakeven_hit_rate": INTERCEPT_COST_MIN / BASELINE_COST_MIN,
    }


def threshold_sensitivity(
    clean_df: pd.DataFrame,
    baseline_costs: list[float] | None = None,
    intercept_costs: list[float] | None = None,
) -> pd.DataFrame:
    """Best threshold and its net saving for a grid of cost assumptions."""
    baseline_costs = baseline_costs or [45, 60, 75, 90, 120]
    intercept_costs = intercept_costs or [5, 8, 12, 20, 30]
    rows = []
    for b in baseline_costs:
        for i in intercept_costs:
            sweep = threshold_sweep(clean_df, baseline_cost=b, intercept_cost=i)
            best = sweep.loc[sweep["net_savings_min"].idxmax()]
            rows.append(
                {
                    "baseline_cost_min": b,
                    "intercept_cost_min": i,
                    "best_threshold": best["threshold"],
                    "intercepted": best["intercepted"],
                    "true_hits": best["true_hits"],
                    "hit_rate": best["hit_rate"],
                    "net_savings_min": best["net_savings_min"],
                    "savings_vs_baseline": best["savings_vs_baseline"],
                }
            )
    return pd.DataFrame(rows)


def fixed_threshold_sensitivity(
    clean_df: pd.DataFrame,
    threshold: float = 18.0,
    intercept_costs: list[float] | None = None,
) -> pd.DataFrame:
    """Keep T fixed and vary the interception cost (the fragile assumption)."""
    intercept_costs = intercept_costs or [6, 9, 12, 15, 18, 24, 30]
    rows = []
    for i in intercept_costs:
        r = evaluate_threshold(clean_df, threshold, intercept_cost=i)
        rows.append(
            {
                "threshold": threshold,
                "intercept_cost_min": i,
                "intercepted": r.intercepted,
                "true_hits": r.true_hits,
                "hit_rate": r.hit_rate,
                "net_savings_min": r.net_savings_min,
                "savings_vs_baseline": r.savings_vs_baseline,
                "clears_costs": r.net_savings_min > 0,
            }
        )
    return pd.DataFrame(rows)
