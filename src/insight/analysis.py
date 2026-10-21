"""Secondary analyses that make the interception rule defensible.

* ``segment_table``       - where the stoppages are concentrated (workshop,
                            category) and whether the rate really differs.
* ``value_at_risk``       - absolute stoppage minutes per group, because a
                            high rate in a 20-row group is not worth a rule.
* ``predictor_comparison``- AUC of ledger lag vs recorded stock vs queue length,
                            so "use delay" is a measured choice, not a hunch.
* ``holdout_check``       - pick T on the first N days, score it on the rest.
* ``ablation``            - does lag add signal on top of stock/queue, and does
                            stock/queue add signal on top of lag?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as m

MIN_GROUP_N = 20
STOP_COST_MIN = m.BASELINE_COST_MIN


# ------------------------------------------------------------------ ranking
def rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    """AUC = P(score of a stoppage > score of a success) via the Mann-Whitney U."""
    pos = scores[labels == 0]
    neg = scores[labels == 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    u = ranks[labels == 0].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def segment_table(df: pd.DataFrame, by: str, min_n: int = MIN_GROUP_N) -> pd.DataFrame:
    """Missing rate, Wilson CI and absolute downtime for one grouping column.

    ``min_n`` gates the rate's usefulness: below it the CI is wide enough that
    the rate should not drive a per-segment threshold.
    """
    rows = []
    for name, g in df.groupby(by):
        n = len(g)
        k = int((g["assembled"] == 0).sum())
        lo, hi = m.wilson_interval(k, n)
        rows.append(
            {
                by: name,
                "n": n,
                "stoppages": k,
                "missing_rate": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "stoppage_minutes": k * STOP_COST_MIN,
                "ci_width": hi - lo,
                "sample_ok_for_threshold": n >= min_n,
            }
        )
    out = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    return out.reset_index(drop=True)


def segment_test(df: pd.DataFrame, by: str) -> dict:
    """Chi-square test of independence between a grouping column and outcome."""
    tab = pd.crosstab(df[by], df["assembled"])
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"by": by, "chi2": 0.0, "p_value": 1.0, "dof": 0, "cramers_v": 0.0}
    table = tab.to_numpy()
    chi2, p, dof = m.chi2_p_value(table)
    return {
        "by": by,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": m.cramers_v(table),
    }


def value_at_risk(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Absolute downtime per segment, split by lag band (captures both)."""
    df = df.copy()
    df["lag_band"] = np.where(df["delay_min"] >= 20, ">=20", "<20")
    rows = []
    for (seg, band), g in df.groupby([by, "lag_band"], observed=True):
        k = int((g["assembled"] == 0).sum())
        rows.append(
            {
                by: seg,
                "lag_band": band,
                "n": len(g),
                "stoppages": k,
                "stoppage_minutes": k * STOP_COST_MIN,
            }
        )
    out = pd.DataFrame(rows)
    tot = out.groupby(by)[["stoppages", "stoppage_minutes"]].sum()
    out = out.merge(
        tot.rename(columns={"stoppages": "segment_stoppages", "stoppage_minutes": "segment_minutes"}),
        left_on=by,
        right_index=True,
    )
    out["share_of_segment_minutes"] = out["stoppage_minutes"] / out["segment_minutes"]
    return out.sort_values([by, "lag_band"]).reset_index(drop=True)


# ---------------------------------------------------------------- predictors
def predictor_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """AUC for each single predictor of a stoppage."""
    specs = [
        ("delay_min", "ledger lag (min)"),
        ("recorded_stock", "recorded stock"),
        ("queue_length", "queue length"),
    ]
    rows = []
    y = (df["assembled"] == 0).astype(int)
    for col, label in specs:
        rows.append(
            {
                "predictor": col,
                "label": label,
                "auc": rank_auc(df[col].astype(float), y),
                "auc_inverted": rank_auc(-df[col].astype(float), y),
            }
        )
    out = pd.DataFrame(rows)
    out["best_auc"] = out[["auc", "auc_inverted"]].max(axis=1)
    return out.sort_values("best_auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- holdout
def workshop_holdout(clean_df: pd.DataFrame, global_threshold: float) -> pd.DataFrame:
    """Does a workshop-specific threshold survive a split-half test?

    For each workshop: sort by time, take the earliest half to pick the local
    threshold, then score that local threshold and the global one on the later
    half only.  A local threshold is worth deploying only if it beats the global
    rule *out of sample* by a margin that is larger than its noise.
    """
    rows = []
    for ws, g in clean_df.groupby("workshop"):
        g = g.sort_values("request_time").reset_index(drop=True)
        half = len(g) // 2
        first, second = g.iloc[:half], g.iloc[half:]
        local = m.optimal_threshold(m.threshold_sweep(first))
        local_test = m.evaluate_threshold(second, local["threshold"])
        global_test = m.evaluate_threshold(second, global_threshold)
        rows.append(
            {
                "workshop": ws,
                "n": len(g),
                "train_n": len(first),
                "test_n": len(second),
                "min_train_threshold": local["threshold"],
                "min_train_net_min": local["net_savings_min"],
                "global_threshold": global_threshold,
                "test_local_net_min": local_test.net_savings_min,
                "test_global_net_min": global_test.net_savings_min,
                "test_delta_local_minus_global": local_test.net_savings_min - global_test.net_savings_min,
                "local_wins_out_of_sample": local_test.net_savings_min > global_test.net_savings_min,
            }
        )
    return pd.DataFrame(rows)


def holdout_check(    df: pd.DataFrame,
    train_days: int = 4,
    thresholds: list[float] | None = None,
) -> dict:
    """Train the threshold on the first ``train_days``, score it on the rest."""
    days = (df["request_time"] - df["request_time"].min()).dt.total_seconds() / 86400.0
    train = df[days < train_days]
    test = df[days >= train_days]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("holdout split produced an empty side")
    train_sweep = m.threshold_sweep(train, thresholds)
    best = m.optimal_threshold(train_sweep)
    test_res = m.evaluate_threshold(test, best["threshold"])
    test_baseline = int((test["assembled"] == 0).sum()) * m.BASELINE_COST_MIN
    # the honest counterfactual: no rule at all in the test window
    return {
        "train_rows": len(train),
        "train_days": float(train_days),
        "test_rows": len(test),
        "test_days": float((df["request_time"].max() - test["request_time"].min()).days),
        "chosen_threshold": best["threshold"],
        "train_net_savings_min": best["net_savings_min"],
        "train_savings_vs_baseline": best["savings_vs_baseline"],
        "test_intercepted": test_res.intercepted,
        "test_true_hits": test_res.true_hits,
        "test_hit_rate": test_res.hit_rate,
        "test_precision_ci": test_res.precision_ci,
        "test_net_savings_min": test_res.net_savings_min,
        "test_baseline_min": test_baseline,
        "test_savings_vs_baseline": test_res.savings_vs_baseline,
        "test_recall": test_res.recall,
    }


# ---------------------------------------------------------------- ablation
def ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare rules built on lag, stock, queue and their combinations.

    A rule is "intercept if <predictor> crosses <cut>"; the best cut per rule is
    chosen by net savings.  ``or_combined`` flags a row when *either* the lag
    rule or the low-stock rule fires.  Cuts are reported in the natural direction
    ("lag >= T", "stock <= S") so the table reads like the deployment rule.
    """
    rows = []

    def best_rule(scores: pd.Series, label: str, direction: str, to_cut=None) -> dict:
        sweep = []
        for cut in np.unique(np.round(scores, 2)):
            flag = scores >= cut
            n_int = int(flag.sum())
            hits = int(((df["assembled"] == 0) & flag).sum())
            net = hits * m.BASELINE_COST_MIN - n_int * m.INTERCEPT_COST_MIN
            sweep.append((cut, n_int, hits, net))
        sweep = pd.DataFrame(sweep, columns=["cut_score", "intercepted", "hits", "net"])
        b = sweep.loc[sweep["net"].idxmax()]
        cut = float(b["cut_score"])
        return {
            "rule": label,
            "cut": to_cut(cut) if to_cut else cut,
            "cut_direction": direction,
            "intercepted": int(b["intercepted"]),
            "true_hits": int(b["hits"]),
            "hit_rate": float(b["hits"] / b["intercepted"]) if b["intercepted"] else 0.0,
            "net_savings_min": float(b["net"]),
        }

    lag = best_rule(df["delay_min"].astype(float), "ledger lag >= T", ">=")
    stock = best_rule(
        -df["recorded_stock"].astype(float), "stock <= S", "<=", to_cut=lambda c: -c
    )
    rows.extend([lag, stock])
    rows.append(best_rule(df["queue_length"].astype(float), "queue >= Q", ">="))

    # OR of the two best single rules, each tuned on its own
    flag = (df["delay_min"] >= lag["cut"]) | (df["recorded_stock"] <= stock["cut"])
    n_int = int(flag.sum())
    hits = int(((df["assembled"] == 0) & flag).sum())
    rows.append(
        {
            "rule": "lag >= T OR stock <= S",
            "cut": float("nan"),
            "cut_direction": "either",
            "intercepted": n_int,
            "true_hits": hits,
            "hit_rate": hits / n_int if n_int else 0.0,
            "net_savings_min": hits * m.BASELINE_COST_MIN - n_int * m.INTERCEPT_COST_MIN,
        }
    )
    out = pd.DataFrame(rows)
    out["savings_vs_baseline"] = out["net_savings_min"] / (
        int((df["assembled"] == 0).sum()) * m.BASELINE_COST_MIN
    )
    return out
