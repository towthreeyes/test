"""Orchestration: run every analysis, write tables/figures/JSON, print a digest."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import pandas as pd

from . import analysis as a
from . import charts
from . import metrics as m
from .data import load

OUT_DIRS = ("outputs/tables", "outputs/figures")


def ensure_dirs() -> None:
    for d in ("outputs", "outputs/tables", "outputs/figures"):
        os.makedirs(d, exist_ok=True)


def _write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(f"outputs/tables/{name}.csv", index=False, encoding="utf-8-sig")


def _round(v: object, digits: int = 4):
    if isinstance(v, float):
        return round(v, digits)
    if isinstance(v, tuple):
        return tuple(round(x, digits) for x in v)
    return v


def main(threshold: float = 19.0) -> dict:
    ensure_dirs()
    df, raw_report = load()
    clean = m.clean(df)
    summary = m.clean_summary(df, clean)
    y = (clean["assembled"] == 0).astype(int)

    # ---------------------------------------------------------- 1. curve
    curve = m.delay_bucket_curve(clean)
    _write_csv(curve, "delay_bucket_curve")

    # ---------------------------------------------------------- 2. tests
    tests = []
    for t in (10, 15, 18, 19, 20, 25, 30, 40, 60):
        r = m.two_group_test(clean, t)
        tests.append(
            {
                "threshold": t,
                "n_high": r["n_high"],
                "stoppages_high": r["stoppages_high"],
                "rate_high": r["rate_high"],
                "ci_high": r["ci_high"],
                "n_low": r["n_low"],
                "stoppages_low": r["stoppages_low"],
                "rate_low": r["rate_low"],
                "ci_low": r["ci_low"],
                "rate_ratio": r["rate_ratio"],
                "odds_ratio": r["odds_ratio"],
                "chi2": r["chi2"],
                "p_value": r["p_value"],
                "cramers_v": r["cramers_v"],
            }
        )
    tests_df = pd.DataFrame(tests)
    _write_csv(tests_df, "threshold_tests")

    # ---------------------------------------------------------- 3. sweep
    sweep = m.threshold_sweep(clean)
    _write_csv(sweep, "threshold_sweep")
    best = m.optimal_threshold(sweep)
    chosen = m.evaluate_threshold(clean, threshold)

    # ---------------------------------------------------------- 4. fixed T sensitivity
    fx = m.fixed_threshold_sensitivity(clean, threshold=best["threshold"])
    _write_csv(fx, "intercept_cost_sensitivity")

    # ---------------------------------------------------------- 5. segments
    segments = {}
    for by in ("workshop", "part_category"):
        seg = a.segment_table(clean, by)
        _write_csv(seg, f"segment_{by}")
        var = a.value_at_risk(clean, by)
        _write_csv(var, f"value_at_risk_{by}")
        segments[by] = {
            "table": seg.to_dict(orient="records"),
            "test": a.segment_test(clean, by),
            "value_at_risk": var.to_dict(orient="records"),
        }

    # per-workshop optimum, restricted to that workshop only.  This is an
    # in-sample fit; the split-half check below is what decides whether a local
    # threshold is worth deploying.
    workshop_opt = []
    for ws, g in clean.groupby("workshop"):
        sw = m.threshold_sweep(g)
        b = m.optimal_threshold(sw)
        base = int((g["assembled"] == 0).sum()) * m.BASELINE_COST_MIN
        chosen_ws = m.evaluate_threshold(g, b["threshold"])
        rows_at_global = m.evaluate_threshold(g, threshold)
        workshop_opt.append(
            {
                "workshop": ws,
                "n": len(g),
                "stoppages": int((g["assembled"] == 0).sum()),
                "missing_rate": float((g["assembled"] == 0).mean()),
                "in_sample_best_threshold": b["threshold"],
                "in_sample_best_net_min": b["net_savings_min"],
                "global_threshold": threshold,
                "global_net_min": rows_at_global.net_savings_min,
                "in_sample_gain_min": b["net_savings_min"] - rows_at_global.net_savings_min,
                "local_rule_loses_money": b["net_savings_min"] <= 0,
                "at_global_intercepted": rows_at_global.intercepted,
                "at_global_hits": rows_at_global.true_hits,
                "at_global_hit_rate": rows_at_global.hit_rate,
                "baseline_min": base,
            }
        )
    workshop_opt_df = pd.DataFrame(workshop_opt).sort_values("missing_rate", ascending=False)
    _write_csv(workshop_opt_df, "workshop_threshold_opt")

    # split-half: pick the local T on the first half, score it on the second
    workshop_holdout = a.workshop_holdout(clean, threshold)
    _write_csv(workshop_holdout, "workshop_split_half")

    # ---------------------------------------------------------- 6. predictors / ablation
    preds = a.predictor_comparison(clean)
    _write_csv(preds, "predictor_comparison")
    abl = a.ablation(clean)
    _write_csv(abl, "ablation")

    # ---------------------------------------------------------- 7. holdout
    hold = a.holdout_check(clean)

    # ---------------------------------------------------------- 8. raw-vs-clean reconciliation
    recon = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": raw_report.rows},
            {"metric": "duplicate_rows_removed", "value": summary["duplicate_rows_removed"]},
            {"metric": "missing_outcome_rows", "value": summary["missing_outcome_rows"]},
            {"metric": "clean_rows", "value": summary["clean_rows"]},
            {"metric": "stoppages", "value": summary["clean_stoppages"]},
            {
                "metric": "missing_rate_user_denominator",
                "value": summary["raw_missing_rate_incl_missing_as_denominator"],
            },
            {
                "metric": "missing_rate_valid_denominator",
                "value": summary["raw_missing_rate_valid_denominator"],
            },
            {"metric": "missing_rate_clean", "value": summary["clean_missing_rate"]},
        ]
    )
    _write_csv(recon, "validation_reconciliation")

    # ---------------------------------------------------------- figures
    labels = [str(x) for x in curve["delay_bucket"]]
    values = curve["missing_rate"].tolist()
    cis = list(zip(curve["ci_low"], curve["ci_high"]))
    colors = [charts.ACCENT if float(x.split("-")[0]) >= 20 else charts.NEUTRAL for x in labels]
    charts.save(
        "outputs/figures/fig1_delay_curve.svg",
        charts.bar_chart(
            labels,
            values,
            cis,
            title="台账时滞一过 20 分钟，缺件率从 4% 跳到 31–38%",
            subtitle=(
                f"n={len(clean)} 条有效派工；灰色=20 分钟以下，绿色=20 分钟及以上；"
                "误差线为 Wilson 95% 置信区间"
            ),
            y_label="缺件率",
            bar_colors=colors,
        ),
    )
    sxs = sweep["threshold"].tolist()
    sys_ = sweep["net_savings_min"].tolist()
    charts.save(
        "outputs/figures/fig2_threshold_sweep.svg",
        charts.line_chart(
            sxs,
            sys_,
            title=f"拦截阈值与净节省的关系：T={best['threshold']:g} 分钟是收益峰值",
            subtitle=(
                f"净节省 = 真命中×{m.BASELINE_COST_MIN:g} − 拦截单数×{m.INTERCEPT_COST_MIN:g} 分钟；"
                f"最优 {best['net_savings_min']:,.0f} 分钟 = 基线损失的 {best['savings_vs_baseline']:.1%}"
            ),
            x_label="拦截阈值 T（台账时滞，分钟）",
            y_label="净节省（分钟）",
            highlight_x=best["threshold"],
            highlight_label=f"T={best['threshold']:g}",
            x_min=0,
            x_max=70,
            x_tick_step=10,
        ),
    )
    ws = segments["workshop"]["table"]
    charts.save(
        "outputs/figures/fig3_workshop.svg",
        charts.bar_chart(
            [r["workshop"] for r in ws],
            [r["missing_rate"] for r in ws],
            [(r["ci_low"], r["ci_high"]) for r in ws],
            title="车间之间缺件率差异真实存在：三号精密 15.2% vs 一号冲压 6.9%",
            subtitle="χ² 检验 p=0.0096；误差线为 Wilson 95% 置信区间",
            y_label="缺件率",
        ),
    )

    # ---------------------------------------------------------- JSON bundle
    bundle = {
        "data": {
            "raw": asdict(raw_report),
            "clean": summary,
        },
        "headline": {
            "threshold": best["threshold"],
            "intercepted": best["intercepted"],
            "true_hits": best["true_hits"],
            "false_alarms": best["false_alarms"],
            "hit_rate": best["hit_rate"],
            "hit_rate_ci": chosen.precision_ci,
            "recall": chosen.recall,
            "net_savings_min": best["net_savings_min"],
            "baseline_min": summary["clean_stoppages"] * m.BASELINE_COST_MIN,
            "savings_vs_baseline": best["savings_vs_baseline"],
            "breakeven_hit_rate": best["breakeven_hit_rate"],
            "plateau_min": best["plateau_min"],
            "plateau_max": best["plateau_max"],
            "baseline_cost_min": m.BASELINE_COST_MIN,
            "intercept_cost_min": m.INTERCEPT_COST_MIN,
        },
        "chosen_threshold": {
            "threshold": threshold,
            "intercepted": chosen.intercepted,
            "true_hits": chosen.true_hits,
            "false_alarms": chosen.false_alarms,
            "hit_rate": chosen.hit_rate,
            "recall": chosen.recall,
            "net_savings_min": chosen.net_savings_min,
            "savings_vs_baseline": chosen.savings_vs_baseline,
        },
        "tests": {f"T={t}": tests_df[tests_df["threshold"] == t].iloc[0].to_dict() for t in (18, 20)},
        "segments": {
            by: {
                "table": segments[by]["table"],
                "test": segments[by]["test"],
            }
            for by in segments
        },
        "workshop_opt": workshop_opt_df.to_dict(orient="records"),
        "workshop_split_half": workshop_holdout.to_dict(orient="records"),
        "predictors": preds.to_dict(orient="records"),
        "ablation": abl.to_dict(orient="records"),
        "holdout": hold,
        "curve": curve.to_dict(orient="records"),
        "cost_sensitivity": fx.to_dict(orient="records"),
    }
    with open("outputs/results.json", "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, ensure_ascii=False, indent=2, default=_round)

    _print_digest(bundle)
    return bundle


def _print_digest(b: dict) -> None:
    h = b["headline"]
    d = b["data"]["clean"]
    print("=" * 78)
    print("DATA")
    print(f"  raw rows                 {b['data']['raw']['rows']}")
    print(f"  duplicates removed       {d['duplicate_rows_removed']}")
    print(f"  missing outcome removed  {d['missing_outcome_rows']}")
    print(f"  clean rows               {d['clean_rows']}   stoppages {d['clean_stoppages']}")
    print(f"  missing rate (clean)     {d['clean_missing_rate']:.2%}")
    print(f"  ledger lag mean / p50    {d['delay_mean']:.1f} / {d['delay_p50']:.0f} min")
    print("=" * 78)
    print("CURVE")
    for r in b["curve"]:
        print(
            f"  {r['delay_bucket']:>7s} min  n={r['n']:>4d}  rate={r['missing_rate']:.3f}"
            f"  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )
    print("=" * 78)
    print("HEADLINE RULE")
    print(f"  best T                   {h['threshold']:g} min (plateau {h['plateau_min']:g}-{h['plateau_max']:g})")
    print(f"  intercepted / hits       {h['intercepted']} / {h['true_hits']}")
    print(f"  hit rate                 {h['hit_rate']:.2%}  (breakeven {h['breakeven_hit_rate']:.1%})")
    print(f"  recall                   {h['recall']:.2%}")
    print(f"  net saving               {h['net_savings_min']:,.0f} min of {h['baseline_min']:,.0f}"
          f"  = {h['savings_vs_baseline']:.1%}")
    print("=" * 78)
    print("HOLDOUT")
    ho = b["holdout"]
    print(f"  train {ho['train_rows']} rows -> chose T={ho['chosen_threshold']:g}")
    print(f"  test  {ho['test_rows']} rows -> hit {ho['test_hit_rate']:.2%}, "
          f"net {ho['test_net_savings_min']:,.0f} min = {ho['test_savings_vs_baseline']:.1%} of baseline")
    print("=" * 78)
    print("PREDICTORS (AUC)")
    for r in b["predictors"]:
        print(f"  {r['label']:>16s}  {r['best_auc']:.3f}")
    print("=" * 78)
    print("WORKSHOP OPTIMUM (in-sample)")
    for r in b["workshop_opt"]:
        print(
            f"  {r['workshop']:<12s} n={r['n']:>3d} rate={r['missing_rate']:.2%} "
            f"local T={r['in_sample_best_threshold']:g} net {r['in_sample_best_net_min']:,.0f} "
            f"vs global {r['global_net_min']:,.0f}  "
            f"{'LOSES MONEY' if r['local_rule_loses_money'] else ''}"
        )
    print("-" * 78)
    print("WORKSHOP SPLIT-HALF (pick on first half, score on second)")
    for r in b["workshop_split_half"]:
        print(
            f"  {r['workshop']:<12s} train T={r['min_train_threshold']:g} "
            f"test local {r['test_local_net_min']:,.0f} vs global {r['test_global_net_min']:,.0f} "
            f"-> {'local wins' if r['local_wins_out_of_sample'] else 'global wins'}"
        )
    print("=" * 78)
