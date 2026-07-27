"""
example_three_scenarios.py
==========================
Demo of the standard 3-scenario stress test:

    Baseline -> NGFS 'Baseline'
    Adverse  -> NGFS 'Delayed transition'
    Severe   -> NGFS 'Fragmented World'

Reuses the synthetic historical macro+PD frame from `example_usage.py`.
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macro_selection_engine import EngineConfig, run_three_scenarios
from macro_selection_engine.example_usage import make_synthetic_historical


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ngfs_path = "/mnt/user-data/uploads/ngfs_nigem.xlsx"
    hist_path = os.path.join(repo, "synthetic_historical.csv")
    out_path  = os.path.join(repo, "stress_test_3scenarios.json")

    # Build / refresh synthetic historical
    make_synthetic_historical(n_years=20, start_year=2005, seed=42) \
        .to_csv(hist_path, index=False)

    cfg = EngineConfig(
        country="France",
        # cfg.scenario is IGNORED by run_three_scenarios; kept for clarity only
        scenario="(unused)",
        baseline_scenario="Baseline",
        risk_channel="combined",         # used for Adverse and Severe
        target_variable="PD",
        risk_type="credit",
        ngfs_path=ngfs_path,
        historical_path=hist_path,
        output_path=out_path,
        # tuning for the small synthetic sample
        s1_keep_top_variance_pct=0.80,
        min_abs_correlation=0.15,
        vif_threshold=10.0,
        pvalue_threshold=0.20,
        max_combo_size=3,
        max_candidate_vars=6,
        top_n_models=5,
    )

    out = run_three_scenarios(cfg)

    # ---------------------- pretty summary ----------------------
    print("\n" + "=" * 78)
    print(" 3-SCENARIO STRESS TEST - SUMMARY")
    print("=" * 78)

    md = out["metadata"]
    sm = md["scenario_map"]
    print(f"Country         : {md['country']}  (region: {md['resolved_region']})")
    print(f"Risk type       : {md['risk_type']}   |   target: {md['target_variable']}")
    print(f"Risk channel    : {md['risk_channel_for_stressed']} (for Adverse / Severe)")
    print(f"Scenario map    : Baseline -> {sm['baseline']}")
    print(f"                  Adverse  -> {sm['adverse']}")
    print(f"                  Severe   -> {sm['severe']}")

    print("\n--- Stage 1 (NGFS-only) per stressed scenario ---")
    for tag in ("adverse", "severe"):
        s1 = out["stage1_per_scenario"][tag]
        print(f"  {tag:8s} ({s1['ngfs_scenario']}): {s1['n_selected']} variables")
    print(f"  union  : {len(out['stage1_union'])} variables")

    print("\n--- Stage 2 (historical validation) - best model ---")
    bm = out["best_model"]
    if bm:
        print(f"  name   : {bm['name']}")
        print(f"  family : {bm['family']}")
        print(f"  R2/pR2 : {bm['r2_or_pseudo']:.3f}")
        print(f"  BIC    : {bm['bic']:.2f}")
        print("  coefficients (variable : estimate [se]  p-value):")
        for var, est in bm["coefficients"].items():
            se = bm["std_errors"].get(var, float("nan"))
            pv = bm["pvalues"].get(var, float("nan"))
            print(f"    {var:25s}  {est:+.4f}  [{se:.4f}]  p={pv:.3f}")

    print("\n--- Top 5 satellite models (composite ranking) ---")
    for i, row in enumerate(out["ranking_table"][:5], start=1):
        print(f"  #{i}  score={row['composite_score']:.3f}  "
              f"family={row['family']:<12} "
              f"R2/pR2={row['r2_or_pseudo']:.3f}  BIC={row['bic']:.2f}  "
              f"vars=[{row['variables']}]")

    print("\n--- Projected stressed PD paths (NGFS horizon, annual) ---")
    print(f"  {'Year':>5}   {'Baseline':>10}   {'Adverse':>10}   {'Severe':>10}")

    # Build a year-aligned table from the three projection paths
    paths = {tag: {pt["year"]: pt["value"]
                   for pt in out["projections"][tag]["path"]}
             for tag in ("baseline", "adverse", "severe")}
    all_years = sorted(set().union(*(p.keys() for p in paths.values())))
    # show every 4 years to keep it compact
    for yr in all_years[::4] + ([all_years[-1]] if all_years[-1] not in all_years[::4] else []):
        b = paths["baseline"].get(yr)
        a = paths["adverse"].get(yr)
        s = paths["severe"].get(yr)
        fmt = lambda v: f"{v:.4f}" if v is not None else "  n/a "
        print(f"  {yr:>5}   {fmt(b):>10}   {fmt(a):>10}   {fmt(s):>10}")

    print(f"\nFull JSON output -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
