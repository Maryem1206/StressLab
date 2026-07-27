"""
example_usage.py
================
End-to-end demonstration:

  - NGFS data: the REAL NGFS NiGEM Phase 5 file (provided by the user).
  - Historical data: SYNTHETICALLY GENERATED (annual macro vars + PD),
    consistent with the NGFS variable universe so the mapping has work to do.

The synthetic PD is built with deliberate, realistic dependencies so the
engine can recover something meaningful:
    PD_t = sigmoid( a + b1*Unemployment_t - b2*GDP_growth_t + b3*PolicyRate_t
                    + small noise )
"""

from __future__ import annotations

import os
import sys
import warnings

# silence harmless statsmodels FutureWarnings and benign scipy numdiff warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*invalid value encountered.*")

import numpy as np
import pandas as pd

# allow `python example_usage.py` from inside the package folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macro_selection_engine import EngineConfig, run_engine


def make_synthetic_historical(n_years: int = 20,
                              start_year: int = 2005,
                              seed: int = 42) -> pd.DataFrame:
    """Build a synthetic historical wide frame with annual frequency.

    NOTE on units: this synthetic historical is generated in LEVELS that are
    unit-compatible with the NGFS Baseline (which is in absolute levels:
    e.g. GDP in Bn US$ PPP, unemployment in %, policy rate in %, ...).
    This is critical for the multi-scenario projection step, which adds
    NGFS deviations on top of a historical level anchor.

    Columns:
        year, gdp_level, cpi_inflation, unemployment_rate, policy_rate,
        long_yield, equity_index, exchange_rate, oil_price, coal_price,
        gas_price, carbon_price_proxy, domestic_demand, exports, PD
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)

    def ar1(rho, sigma, n, mean=0.0):
        x = np.zeros(n)
        x[0] = rng.normal(mean, sigma)
        for t in range(1, n):
            x[t] = mean * (1 - rho) + rho * x[t - 1] + rng.normal(0, sigma)
        return x

    # Build a GDP LEVEL trajectory (Bn US$ PPP), starting around 2200 in 2005
    # and ending around 3000 in 2024, with realistic year-on-year growth.
    gdp_growth_rate = ar1(0.5, 1.5, n_years, mean=2.0)  # % growth
    gdp_level = 2200.0 * np.cumprod(1.0 + gdp_growth_rate / 100.0)

    inflation    = ar1(0.7, 0.8, n_years, mean=2.0)
    unemployment = ar1(0.8, 0.6, n_years, mean=8.0).clip(3.5, 14.0)
    policy_rate  = ar1(0.85, 0.5, n_years, mean=2.5).clip(0.0, 8.0)
    long_yield   = policy_rate + ar1(0.6, 0.3, n_years, mean=1.0)
    equity_idx   = 100 * np.exp(np.cumsum(rng.normal(0.04, 0.15, n_years)))
    fx_rate      = 100 + np.cumsum(rng.normal(0.0, 1.5, n_years))
    oil_price    = 60 + np.cumsum(rng.normal(0.0, 6.0, n_years)).clip(-30, 60)
    coal_price   = 80 + np.cumsum(rng.normal(0.0, 5.0, n_years)).clip(-30, 60)
    gas_price    = 50 + np.cumsum(rng.normal(0.0, 4.0, n_years)).clip(-25, 50)
    carbon_proxy = 25 + np.cumsum(rng.normal(0.5, 2.0, n_years)).clip(-10, 80)
    # domestic demand and exports also as LEVELS (Bn US$ PPP)
    dom_demand   = 0.8 * gdp_level + rng.normal(0, 30, n_years)
    exports      = 0.3 * gdp_level + rng.normal(0, 20, n_years)

    # PD depends on RATES (growth, unemployment, policy_rate), not levels —
    # so for the structural relation we use gdp_growth_rate, NOT gdp_level.
    linpred = (-3.5
               + 0.30 * (unemployment - unemployment.mean())
               - 0.25 * (gdp_growth_rate - gdp_growth_rate.mean())
               + 0.10 * (policy_rate - policy_rate.mean())
               + rng.normal(0, 0.15, n_years))
    pd_values = 1.0 / (1.0 + np.exp(-linpred))
    pd_values = np.clip(pd_values, 0.005, 0.30)

    return pd.DataFrame({
        "year": years,
        "gdp_level": gdp_level,                # NGFS-compatible LEVEL
        "gdp_growth": gdp_growth_rate,         # for the satellite-model fit
        "cpi_inflation": inflation,
        "unemployment_rate": unemployment,
        "policy_rate": policy_rate,
        "long_yield": long_yield,
        "equity_index": equity_idx,
        "exchange_rate": fx_rate,
        "oil_price": oil_price,
        "coal_price": coal_price,
        "gas_price": gas_price,
        "carbon_price_proxy": carbon_proxy,
        "domestic_demand": dom_demand,
        "exports": exports,
        "PD": pd_values,
    })


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)

    ngfs_path = "/mnt/user-data/uploads/ngfs_nigem.xlsx"
    hist_path = os.path.join(repo, "synthetic_historical.csv")
    out_path  = os.path.join(repo, "engine_output.json")

    # Build synthetic historical and persist it so the loader path is genuine
    hist_df = make_synthetic_historical(n_years=20, start_year=2005, seed=42)
    hist_df.to_csv(hist_path, index=False)

    cfg = EngineConfig(
        country="France",
        scenario="Net Zero 2050",
        baseline_scenario="Baseline",       # Phase 5: 'Baseline' is the unstressed reference
        risk_channel="combined",            # STRICT
        target_variable="PD",
        risk_type="credit",
        ngfs_path=ngfs_path,
        historical_path=hist_path,
        output_path=out_path,
        # tweak a couple of filters for a small synthetic sample
        s1_keep_top_variance_pct=0.80,
        min_abs_correlation=0.15,
        vif_threshold=10.0,                 # 20-year sample → relax
        pvalue_threshold=0.20,
        max_combo_size=3,
        max_candidate_vars=6,
        top_n_models=5,
    )

    result = run_engine(cfg)

    print("\n" + "=" * 78)
    print("ENGINE SUMMARY")
    print("=" * 78)
    md = result["metadata"]
    print(f"Country/Region : {md['country']}  (resolved: {md['resolved_region']})")
    print(f"Scenario       : {md['scenario']}  vs baseline {md['baseline_scenario']}")
    print(f"Risk channel   : {md['risk_channel']}")
    print(f"Target / risk  : {md['target_variable']}  ({md['risk_type']})")
    print()
    print(f"Stage 1 selected NGFS variables ({result['stage1_ngfs']['n_selected']}):")
    for v in result['stage1_ngfs']['selected_variables']:
        print(f"  - {v}")

    print("\nStage 2 NGFS->historical mapping:")
    for k, v in result['stage2_historical']['mapping']['ngfs_to_historical'].items():
        method = result['stage2_historical']['mapping']['method_per_match'].get(k, "?")
        print(f"  {k}  ->  {v}   [{method}]")
    if result['stage2_historical']['mapping']['unmatched_dropped']:
        print("  Unmatched (dropped):")
        for v in result['stage2_historical']['mapping']['unmatched_dropped']:
            print(f"    - {v}")

    print(f"\nTop {len(result['selected_models'])} models (by composite score):")
    for i, row in enumerate(result['ranking_table'][:5], start=1):
        print(f"  #{i}  score={row['composite_score']:.3f}  "
              f"family={row['family']:<12} "
              f"R2/pR2={row['r2_or_pseudo']:.3f}  BIC={row['bic']:.2f}  "
              f"vars=[{row['variables']}]")

    if result['best_model']:
        bm = result['best_model']
        print("\nBest model:")
        print(f"  name   : {bm['name']}")
        print(f"  family : {bm['family']}")
        print("  coefficients (variable : estimate [se]  p-value):")
        for var, est in bm['coefficients'].items():
            se = bm['std_errors'].get(var, float('nan'))
            pv = bm['pvalues'].get(var, float('nan'))
            print(f"    {var:30s}  {est:+.4f}   [{se:.4f}]   p={pv:.3f}")

    print(f"\nFull JSON output -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
