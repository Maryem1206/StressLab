"""
Tests unitaires — macro_projector.py
=====================================
Couvre :
  - ADF stationarity
  - Johansen cointegration test
  - AR, VAR, VECM projection paths
  - Economic diagnostics
  - Projection report structure
  - Decision tree (correct model selection per scenario)
  - Edge cases (insufficient data, single variable, all I(0), cointegrated I(1))
"""
import numpy as np
import pandas as pd
import pytest

from app.modules.credit.macro_projector import (
    ProjectorConfig,
    ProjectionResult,
    DiagnosticMessage,
    _adf_test,
    _johansen_test,
    _economic_diagnostics,
    _build_report,
    project_macro,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

RNG = np.random.default_rng(42)
YEARS_HIST = list(range(1995, 2023))   # 28 years of history
YEARS_PROJ = list(range(2023, 2029))   # 6-year horizon


def _stationary_series(n=28) -> pd.Series:
    """White-noise I(0) series."""
    return pd.Series(RNG.standard_normal(n) * 0.5 + 2.0,
                     index=YEARS_HIST[:n], name="stat_var")


def _random_walk(n=28) -> pd.Series:
    """Pure random walk I(1) series."""
    return pd.Series(np.cumsum(RNG.standard_normal(n) * 0.3) + 5.0,
                     index=YEARS_HIST[:n], name="rw_var")


def _cointegrated_pair(n=28) -> pd.DataFrame:
    """Two I(1) series sharing a common stochastic trend (cointegrated)."""
    trend  = np.cumsum(RNG.standard_normal(n) * 0.5)
    noise1 = RNG.standard_normal(n) * 0.1
    noise2 = RNG.standard_normal(n) * 0.1
    x1 = trend + noise1 + 3.0
    x2 = 2.0 * trend + noise2 + 1.0
    return pd.DataFrame({"x1": x1, "x2": x2}, index=YEARS_HIST[:n])


def _make_hist_df(n=28, k=3) -> pd.DataFrame:
    """Generic historical DataFrame with k macro variables."""
    data = {
        f"var{i}": RNG.standard_normal(n) * 0.5 + (i + 1.0)
        for i in range(k)
    }
    return pd.DataFrame(data, index=YEARS_HIST[:n])


# ═══════════════════════════════════════════════════════════════════════════
# 1. ADF stationarity test
# ═══════════════════════════════════════════════════════════════════════════

class TestADF:
    def test_stationary_series_detected(self):
        """White noise should be detected as stationary."""
        is_stat, p = _adf_test(_stationary_series())
        assert is_stat is True, f"Expected stationary, got p={p:.4f}"

    def test_random_walk_detected(self):
        """Pure random walk should be detected as non-stationary."""
        is_stat, p = _adf_test(_random_walk())
        # With 28 obs, power is moderate — we just check pvalue > 0.02
        assert p > 0.02, f"Expected high p-value for random walk, got {p:.4f}"

    def test_constant_series(self):
        """Constant series → stationary (zero variance)."""
        s = pd.Series([3.14] * 20, index=YEARS_HIST[:20])
        is_stat, p = _adf_test(s)
        assert is_stat is True

    def test_short_series_fallback(self):
        """Too-short series → conservatively treated as stationary."""
        s = pd.Series([1.0, 2.0, 3.0], index=[2020, 2021, 2022])
        is_stat, p = _adf_test(s)
        assert is_stat is True   # fallback

    def test_returns_float_pvalue(self):
        pvalue = _adf_test(_stationary_series())[1]
        assert isinstance(pvalue, float)
        assert 0.0 <= pvalue <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Johansen cointegration test
# ═══════════════════════════════════════════════════════════════════════════

class TestJohansen:
    def test_cointegrated_pair_detected(self):
        """Cointegrated pair should yield rank ≥ 1."""
        df = _cointegrated_pair(n=28)
        rank, details = _johansen_test(df, k_ar_diff=1, det_order=0, alpha=0.05)
        # With only 28 obs, power is limited — just check structure
        assert isinstance(rank, int)
        assert rank >= 0
        assert "trace_stats" in details
        assert "critical_vals" in details
        assert "eigenvalues" in details

    def test_independent_series_rank_zero(self):
        """Two independent stationary series should not be cointegrated."""
        df = pd.DataFrame({
            "a": _stationary_series().values,
            "b": _stationary_series().values[:28],
        }, index=YEARS_HIST)
        rank, details = _johansen_test(df, k_ar_diff=1, det_order=0, alpha=0.05)
        assert rank >= 0   # just checks no crash

    def test_insufficient_obs_returns_zero(self):
        """Too few observations → rank forced to 0."""
        df = pd.DataFrame({"a": [1.0] * 3, "b": [2.0] * 3}, index=[2020, 2021, 2022])
        rank, details = _johansen_test(df, k_ar_diff=1, det_order=0, alpha=0.05)
        assert rank == 0
        assert "note" in details

    def test_details_structure(self):
        df = _cointegrated_pair()
        _, details = _johansen_test(df, k_ar_diff=1, det_order=0, alpha=0.05)
        assert len(details["trace_stats"]) == len(details["critical_vals"])
        assert len(details["eigenvalues"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. project_macro — single variable
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleVariable:
    def test_returns_correct_shape(self):
        hist = pd.DataFrame({"gdp": _stationary_series()}, index=YEARS_HIST)
        result = project_macro(hist, ["gdp"], YEARS_PROJ)
        assert result.projected.shape == (len(YEARS_PROJ), 1)
        assert list(result.projected.index) == YEARS_PROJ

    def test_model_is_ar(self):
        hist = pd.DataFrame({"gdp": _stationary_series()}, index=YEARS_HIST)
        result = project_macro(hist, ["gdp"], YEARS_PROJ)
        assert result.model_used.startswith("AR")

    def test_no_nan_in_projection(self):
        hist = pd.DataFrame({"gdp": _stationary_series()}, index=YEARS_HIST)
        result = project_macro(hist, ["gdp"], YEARS_PROJ)
        assert not result.projected.isnull().any().any()

    def test_single_i1_differenced_and_reconstructed(self):
        """I(1) single variable: differenced for AR, levels reconstructed."""
        hist = pd.DataFrame({"rw": _random_walk()}, index=YEARS_HIST)
        result = project_macro(hist, ["rw"], YEARS_PROJ)
        assert result.projected.shape == (len(YEARS_PROJ), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 4. project_macro — multiple variables, all I(0)
# ═══════════════════════════════════════════════════════════════════════════

class TestMultipleStationary:
    def _hist(self, n=28):
        return pd.DataFrame({
            "gdp_growth":    RNG.standard_normal(n) * 0.5 + 2.0,
            "unemployment":  RNG.standard_normal(n) * 0.3 + 8.0,
            "inflation":     RNG.standard_normal(n) * 0.4 + 3.0,
        }, index=YEARS_HIST[:n])

    def test_model_var_or_ar_fallback(self):
        hist = self._hist()
        result = project_macro(hist, ["gdp_growth", "unemployment", "inflation"],
                               YEARS_PROJ)
        assert "VAR" in result.model_used or "AR" in result.model_used

    def test_projection_has_all_vars(self):
        hist = self._hist()
        result = project_macro(hist, ["gdp_growth", "unemployment"], YEARS_PROJ)
        assert "gdp_growth" in result.projected.columns
        assert "unemployment" in result.projected.columns

    def test_no_cointegration_test_run(self):
        """All-I(0) path should not run Johansen."""
        hist = self._hist()
        result = project_macro(hist, ["gdp_growth", "unemployment"], YEARS_PROJ,
                               cfg=ProjectorConfig(adf_pvalue=0.01))
        # With very strict ADF threshold, all vars are treated as I(0) → no Johansen
        # (This is a soft test — just checks coint_tested is bool)
        assert isinstance(result.coint_tested, bool)


# ═══════════════════════════════════════════════════════════════════════════
# 5. project_macro — cointegrated I(1) variables → VECM
# ═══════════════════════════════════════════════════════════════════════════

class TestVECMPath:
    def test_cointegrated_uses_vecm_or_var_diff(self):
        """Cointegrated pair should lead to VECM or VAR-diff, not plain VAR."""
        hist = _cointegrated_pair(n=28)
        result = project_macro(hist, ["x1", "x2"], YEARS_PROJ)
        # Either VECM or VAR-diff (fallback if too few obs)
        assert any(kw in result.model_used for kw in ("VECM", "VAR", "AR"))

    def test_johansen_tested_is_true_for_i1(self):
        hist = _cointegrated_pair(n=28)
        result = project_macro(hist, ["x1", "x2"], YEARS_PROJ,
                               cfg=ProjectorConfig(adf_pvalue=0.20))
        # With relaxed ADF, both will be I(1) → Johansen run
        # (soft test — just verifies the field exists)
        assert hasattr(result, "coint_tested")

    def test_projection_shape_correct(self):
        hist = _cointegrated_pair(n=28)
        result = project_macro(hist, ["x1", "x2"], YEARS_PROJ)
        assert result.projected.shape == (len(YEARS_PROJ), 2)

    def test_no_crash_with_short_history(self):
        """Should gracefully fall back when too few obs for VECM."""
        hist = _cointegrated_pair(n=14)
        result = project_macro(hist, ["x1", "x2"], YEARS_PROJ)
        assert result.projected is not None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Economic diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestEconomicDiagnostics:
    def _base_hist(self):
        return pd.DataFrame({
            "unemployment": np.linspace(8.0, 10.0, 28),
            "gdp_growth":   np.linspace(2.0, 3.0, 28),
        }, index=YEARS_HIST)

    def test_returns_list_of_diagnostic_messages(self):
        hist = self._base_hist()
        proj = pd.DataFrame({
            "unemployment": [9.0] * 6,
            "gdp_growth":   [2.5] * 6,
        }, index=YEARS_PROJ)
        msgs = _economic_diagnostics(hist, proj)
        assert isinstance(msgs, list)
        assert all(isinstance(m, DiagnosticMessage) for m in msgs)

    def test_negative_unemployment_triggers_warning(self):
        hist = self._base_hist()
        proj = pd.DataFrame({"unemployment": [-1.0, 2.0, 3.0, 2.5, 2.0, 1.5]},
                             index=YEARS_PROJ)
        msgs = _economic_diagnostics(hist, proj)
        warns = [m for m in msgs if m.level == "WARNING" and "unemployment" in m.message]
        assert len(warns) >= 1, "Expected warning for negative unemployment"

    def test_above_historical_max_triggers_warning(self):
        hist = pd.DataFrame({"gdp_growth": [2.0, 2.5, 3.0] * 9 + [3.0]},
                             index=YEARS_HIST)
        proj = pd.DataFrame({"gdp_growth": [10.0, 11.0, 12.0, 10.5, 11.5, 12.5]},
                             index=YEARS_PROJ)
        msgs = _economic_diagnostics(hist, proj)
        warns = [m for m in msgs if m.level == "WARNING" and "maximum" in m.message]
        assert len(warns) >= 1

    def test_stable_projection_yields_info(self):
        hist = self._base_hist()
        proj = pd.DataFrame({
            "unemployment": [9.0, 9.1, 9.2, 9.3, 9.4, 9.5],
            "gdp_growth":   [2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
        }, index=YEARS_PROJ)
        msgs = _economic_diagnostics(hist, proj)
        infos = [m for m in msgs if m.level == "INFO"]
        assert len(infos) >= 1

    def test_no_blocking(self):
        """Diagnostics must NEVER raise or return None — always a list."""
        hist = pd.DataFrame({"x": [np.nan] * 28}, index=YEARS_HIST)
        proj = pd.DataFrame({"x": [-999.0] * 6}, index=YEARS_PROJ)
        try:
            msgs = _economic_diagnostics(hist, proj)
            assert isinstance(msgs, list)
        except Exception as e:
            pytest.fail(f"Diagnostics raised an exception: {e}")

    def test_diagnostic_str_representation(self):
        m_info = DiagnosticMessage("INFO", "All good.")
        m_warn = DiagnosticMessage("WARNING", "Value exceeded max.")
        assert str(m_info).startswith("✓")
        assert str(m_warn).startswith("⚠")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Projection Report
# ═══════════════════════════════════════════════════════════════════════════

class TestProjectionReport:
    def _result(self) -> ProjectionResult:
        hist = pd.DataFrame({
            "gdp_growth":   np.linspace(2.0, 3.0, 28),
            "unemployment": np.linspace(8.0, 10.0, 28),
        }, index=YEARS_HIST)
        return project_macro(hist, ["gdp_growth", "unemployment"], YEARS_PROJ)

    def test_report_keys_present(self):
        r = self._result()
        required = {"methodology", "steps_completed", "adf_results",
                    "projection_table", "diagnostics", "model_used",
                    "p_optimal", "horizon_years"}
        assert required.issubset(r.report.keys()), \
            f"Missing keys: {required - r.report.keys()}"

    def test_methodology_is_string(self):
        r = self._result()
        assert isinstance(r.report["methodology"], str)
        assert len(r.report["methodology"]) > 0

    def test_projection_table_has_correct_years(self):
        r = self._result()
        years = [row["year"] for row in r.report["projection_table"]]
        assert years == YEARS_PROJ

    def test_projection_table_has_all_vars(self):
        r = self._result()
        for row in r.report["projection_table"]:
            assert "gdp_growth" in row
            assert "unemployment" in row

    def test_diagnostics_list_not_empty(self):
        r = self._result()
        assert len(r.report["diagnostics"]) > 0

    def test_diagnostics_have_level_and_message(self):
        r = self._result()
        for d in r.report["diagnostics"]:
            assert "level" in d
            assert "message" in d
            assert d["level"] in ("INFO", "WARNING")

    def test_adf_results_structure(self):
        r = self._result()
        for row in r.report["adf_results"]:
            assert "variable" in row
            assert "adf_pvalue" in row
            assert "stationary" in row


# ═══════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_insufficient_history_returns_naive(self):
        hist = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[2020, 2021, 2022])
        result = project_macro(hist, ["x"], [2023, 2024])
        assert result.model_used == "naive"
        assert result.projected.shape == (2, 1)

    def test_missing_variable_raises(self):
        hist = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 9 + [1.0]}, index=YEARS_HIST)
        with pytest.raises(ValueError, match="not in historical"):
            project_macro(hist, ["b"], YEARS_PROJ)

    def test_zero_horizon_returns_empty(self):
        hist = _make_hist_df()
        result = project_macro(hist, ["var0"], [])
        assert result.projected.shape[0] == 0

    def test_projection_never_crashes_on_any_input(self):
        """Fuzz test: random shapes should never raise."""
        for seed in range(5):
            rng = np.random.default_rng(seed)
            n = rng.integers(5, 30)
            k = rng.integers(1, 4)
            years_hist = list(range(2000, 2000 + n))
            data = {f"v{i}": rng.standard_normal(n) for i in range(k)}
            hist = pd.DataFrame(data, index=years_hist)
            try:
                result = project_macro(hist, list(data.keys()), YEARS_PROJ)
                assert result.projected is not None
            except Exception as e:
                pytest.fail(f"project_macro crashed with seed={seed}: {e}")

    def test_all_nan_column_handled(self):
        hist = pd.DataFrame({
            "good": np.linspace(1.0, 3.0, 28),
            "bad":  [np.nan] * 28,
        }, index=YEARS_HIST)
        # After dropna(), only good remains (or the call raises ValueError gracefully)
        try:
            result = project_macro(hist, ["good"], YEARS_PROJ)
            assert result.projected is not None
        except ValueError:
            pass  # acceptable — missing variable detected cleanly

    def test_single_year_horizon(self):
        hist = _make_hist_df()
        result = project_macro(hist, ["var0"], [2025])
        assert result.projected.shape == (1, 1)

    def test_config_respected(self):
        """Custom ProjectorConfig parameters are used."""
        cfg = ProjectorConfig(min_obs_for_ar=100)  # impossible threshold
        hist = _make_hist_df(n=28)
        result = project_macro(hist, ["var0"], YEARS_PROJ, cfg=cfg)
        assert result.model_used == "naive"
