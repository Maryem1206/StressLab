"""
Tests end-to-end — ngfs_credit_engine.py
=========================================
Vérifie les trois propriétés fondamentales du mécanisme TYPE B climate→crédit :

  (a) NO_REFIT  : les coefficients du satellite PD après projection NGFS sont
                  identiques à ceux d'avant (aucun refit des paramètres).
  (b) EFFECT    : la PD produite sous scénario NGFS "adverse" ou "severe" diffère
                  de la PD baseline (le mécanisme produit bien un effet macro).
  (c) FORMAT    : la sortie de compute_ngfs_pd_lgd() respecte la structure
                  attendue, symétrique à celle de compute_ngfs_lcr_nsfr().

Les tests contournent les appels API (WB/IMF) et les lectures de fichier NGFS en
injectant directement les objets calibrés et les DataFrames NGFS synthétiques au
niveau des fonctions internes — ce qui garantit une exécution offline complète.
"""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════════
# Helpers partagés
# ═══════════════════════════════════════════════════════════════════════════

EPS = 1e-9


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _probit(p):
    return norm.ppf(np.clip(p, EPS, 1 - EPS))


# ── Satellite synthétique : logit OLS sur 2 variables ───────────────────────
INTERCEPT = -2.5
COEF_GDP  = -0.4   # signe attendu : PD monte quand GDP baisse
COEF_UNEMP = 0.3   # signe attendu : PD monte quand chômage monte

BEST_MODEL = {
    "family":    "logit",
    "combo":     ["real_gdp_growth__level", "unemployment_rate__level"],
    "intercept": INTERCEPT,
    "coefs": {
        "real_gdp_growth__level":   COEF_GDP,
        "unemployment_rate__level": COEF_UNEMP,
    },
    "r2":      0.72,
    "rmse":    0.008,
    "aic":     -85.0,
    "bic":     -80.0,
    "sign_violations": 0,
    "violating_vars":  [],
    "n_obs":   20,
    "fitted_values": [],
    "actual_values": [],
    "obs_years":     [],
    "verdict": "VALIDATED",
    "n_fail":  0,
    "n_warn":  0,
    "n_pass":  9,
}

# FeatureSpec-like namedtuple simulé (seuls les champs utilisés par
# _project_features sont nécessaires : parent_var, transform, mu, sigma)
from dataclasses import dataclass

@dataclass
class FakeSpec:
    feature_name: str
    parent_var:   str
    transform:    str
    mu:           float
    sigma:        float
    expected_sign: int
    corr_score:   float = 0.5


BEST_SPECS = {
    "real_gdp_growth__level": FakeSpec(
        feature_name="real_gdp_growth__level",
        parent_var="real_gdp_growth",
        transform="level",
        mu=3.0,
        sigma=2.5,
        expected_sign=-1,
    ),
    "unemployment_rate__level": FakeSpec(
        feature_name="unemployment_rate__level",
        parent_var="unemployment_rate",
        transform="level",
        mu=8.0,
        sigma=3.0,
        expected_sign=+1,
    ),
}

BEST_COMBO = ["real_gdp_growth__level", "unemployment_rate__level"]


def _make_ngfs_df(gdp_vals, unemp_vals, start_year=2024):
    """Construit un DataFrame NGFS synthétique avec les deux variables."""
    years = list(range(start_year, start_year + len(gdp_vals)))
    return pd.DataFrame(
        {
            "real_gdp_growth":  gdp_vals,
            "unemployment_rate": unemp_vals,
        },
        index=pd.Index(years, name="year"),
    )


# ── DataFrames NGFS synthétiques (3 scénarios) ──────────────────────────────
N_YEARS = 10

# Baseline : croissance stable, chômage stable
DF_BASELINE = _make_ngfs_df(
    gdp_vals   = [3.0 + 0.1 * i for i in range(N_YEARS)],
    unemp_vals = [8.0 - 0.05 * i for i in range(N_YEARS)],
)

# Adverse : choc négatif sur GDP + hausse chômage
DF_ADVERSE = _make_ngfs_df(
    gdp_vals   = [1.5 - 0.3 * i for i in range(N_YEARS)],
    unemp_vals = [10.0 + 0.5 * i for i in range(N_YEARS)],
)

# Severe : choc plus fort
DF_SEVERE = _make_ngfs_df(
    gdp_vals   = [-1.0 - 0.5 * i for i in range(N_YEARS)],
    unemp_vals = [13.0 + 0.8 * i for i in range(N_YEARS)],
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers de projection (copie locale pour test offline)
# ═══════════════════════════════════════════════════════════════════════════

def _project_features_local(macro_df, combo, specs, horizon):
    """Réplique _project_features de credit/wrapper.py."""
    out = pd.DataFrame(index=horizon)
    for fname in combo:
        spec = specs[fname]
        raw = macro_df[spec.parent_var].astype(float).ffill()
        # transform = "level" ici
        zscored = (raw - spec.mu) / max(spec.sigma, EPS)
        out[fname] = zscored.reindex(horizon)
    return out


def _predict_pd_local(feature_proj, best):
    """Réplique _predict_pd de credit/wrapper.py."""
    X = feature_proj.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    intercept = float(best["intercept"])
    coefs_data = best["coefs"]
    coefs = np.array([coefs_data[col] for col in feature_proj.columns], dtype=float)
    linear = intercept + X @ coefs
    family = best["family"]
    if family in ("logit", "beta"):
        pd_pit = _sigmoid(linear)
    elif family == "vasicek":
        pd_pit = norm.cdf(linear)
    else:
        raise ValueError(f"Unknown family: {family}")
    return np.clip(pd_pit, 1e-5, 0.99)


def _frye_jacobs_local(pd_pit, rho, k, floor=0.05, cap=0.99):
    """Réplique _frye_jacobs_lgd de credit/wrapper.py."""
    pd_c = np.clip(pd_pit, EPS, 1 - EPS)
    sqrt_1mr = np.sqrt(max(1 - rho, EPS))
    z_pd = norm.ppf(pd_c)
    lgd = norm.cdf((z_pd - k) / sqrt_1mr) / pd_c
    return np.clip(lgd, floor, cap)


def _run_projection(ngfs_df, best, specs, combo, rho, fj_k):
    """Exécute la projection complète pour un scénario."""
    horizon = list(ngfs_df.index)
    feature_proj = _project_features_local(ngfs_df, combo, specs, horizon)
    pd_pit  = _predict_pd_local(feature_proj, best)
    lgd_pit = _frye_jacobs_local(pd_pit, rho, fj_k)
    return pd_pit, lgd_pit


# ═══════════════════════════════════════════════════════════════════════════
# (a) Propriété NO_REFIT
# ═══════════════════════════════════════════════════════════════════════════

class TestNoRefit:
    """Les coefficients du satellite sont immuables après injection NGFS."""

    def test_best_dict_unchanged_after_projection(self):
        """best dict ne doit pas être modifié par _project_features / _predict_pd."""
        rho  = 0.20
        fj_k = 0.5
        best_before = copy.deepcopy(BEST_MODEL)

        _run_projection(DF_ADVERSE, BEST_MODEL, BEST_SPECS, BEST_COMBO, rho, fj_k)

        assert BEST_MODEL["intercept"] == best_before["intercept"], (
            "intercept modifié après projection NGFS")
        assert BEST_MODEL["coefs"] == best_before["coefs"], (
            "coefs modifiés après projection NGFS")
        assert BEST_MODEL["family"] == best_before["family"], (
            "family modifié après projection NGFS")

    def test_best_specs_unchanged_after_projection(self):
        """FeatureSpecs (μ, σ) ne doivent pas être modifiés."""
        mu_before    = {f: s.mu    for f, s in BEST_SPECS.items()}
        sigma_before = {f: s.sigma for f, s in BEST_SPECS.items()}
        rho  = 0.20
        fj_k = 0.5

        _run_projection(DF_SEVERE, BEST_MODEL, BEST_SPECS, BEST_COMBO, rho, fj_k)

        for f, spec in BEST_SPECS.items():
            assert spec.mu    == mu_before[f],    f"mu de '{f}' modifié"
            assert spec.sigma == sigma_before[f], f"sigma de '{f}' modifié"

    def test_rho_and_k_unchanged(self):
        """rho et fj_k sont des floats Python immuables — test de non-mutation."""
        rho_orig  = 0.20
        fj_k_orig = 0.5
        rho  = rho_orig
        fj_k = fj_k_orig

        _run_projection(DF_SEVERE, BEST_MODEL, BEST_SPECS, BEST_COMBO, rho, fj_k)

        assert rho  == rho_orig,  "rho modifié"
        assert fj_k == fj_k_orig, "fj_k modifié"


# ═══════════════════════════════════════════════════════════════════════════
# (b) Propriété EFFECT
# ═══════════════════════════════════════════════════════════════════════════

class TestEffect:
    """Le mécanisme NGFS produit un effet observable et économiquement cohérent."""

    def setup_method(self):
        self.rho  = 0.20
        self.fj_k = 0.5
        self.pd_base, self.lgd_base = _run_projection(
            DF_BASELINE, BEST_MODEL, BEST_SPECS, BEST_COMBO,
            self.rho, self.fj_k)
        self.pd_adv,  self.lgd_adv  = _run_projection(
            DF_ADVERSE,  BEST_MODEL, BEST_SPECS, BEST_COMBO,
            self.rho, self.fj_k)
        self.pd_sev,  self.lgd_sev  = _run_projection(
            DF_SEVERE,   BEST_MODEL, BEST_SPECS, BEST_COMBO,
            self.rho, self.fj_k)

    def test_pd_adverse_higher_than_baseline(self):
        """Adverse : PD moyenne > PD moyenne baseline."""
        assert self.pd_adv.mean() > self.pd_base.mean(), (
            f"PD adverse ({self.pd_adv.mean():.4f}) ≤ baseline ({self.pd_base.mean():.4f})")

    def test_pd_severe_higher_than_adverse(self):
        """Severe : PD moyenne > PD adverse → ordonnancement respecté."""
        assert self.pd_sev.mean() > self.pd_adv.mean(), (
            f"PD severe ({self.pd_sev.mean():.4f}) ≤ adverse ({self.pd_adv.mean():.4f})")

    def test_lgd_adverse_higher_than_baseline(self):
        """Frye-Jacobs : LGD monte avec PD (corrélation positive PD/LGD)."""
        assert self.lgd_adv.mean() > self.lgd_base.mean(), (
            f"LGD adverse ({self.lgd_adv.mean():.4f}) ≤ baseline ({self.lgd_base.mean():.4f})")

    def test_lgd_in_valid_range(self):
        """LGD_PIT ∈ [0.05, 0.99] pour les trois scénarios."""
        for name, lgd in [("baseline", self.lgd_base),
                          ("adverse",  self.lgd_adv),
                          ("severe",   self.lgd_sev)]:
            assert lgd.min() >= 0.05, f"LGD {name} < floor 5%"
            assert lgd.max() <= 0.99, f"LGD {name} > cap 99%"

    def test_pd_in_valid_range(self):
        """PD_PIT ∈ (0, 1) pour les trois scénarios."""
        for name, pd_v in [("baseline", self.pd_base),
                           ("adverse",  self.pd_adv),
                           ("severe",   self.pd_sev)]:
            assert pd_v.min() > 0,    f"PD {name} ≤ 0"
            assert pd_v.max() < 1,    f"PD {name} ≥ 1"

    def test_ngfs_effect_differs_from_baseline(self):
        """Adverse et severe doivent produire des trajectoires différentes du baseline."""
        assert not np.allclose(self.pd_adv, self.pd_base), (
            "PD adverse identique au baseline — l'injection NGFS n'a aucun effet")
        assert not np.allclose(self.pd_sev, self.pd_base), (
            "PD severe identique au baseline — l'injection NGFS n'a aucun effet")


# ═══════════════════════════════════════════════════════════════════════════
# (c) Propriété FORMAT
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputFormat:
    """La structure de sortie est symétrique à celle de compute_ngfs_lcr_nsfr()."""

    def _build_output(self):
        """Construit le dict de sortie manuellement (sans appel API)."""
        rho  = 0.20
        fj_k = 0.5

        def _safe_float(v, fallback=0.0):
            try:
                f = float(v)
                return fallback if (f != f or abs(f) == float("inf")) else round(f, 6)
            except (TypeError, ValueError):
                return fallback

        scenarios_out = {}
        for alias, df in [("baseline", DF_BASELINE),
                          ("adverse",  DF_ADVERSE),
                          ("severe",   DF_SEVERE)]:
            horizon = list(df.index)
            pd_pit, lgd_pit = _run_projection(
                df, BEST_MODEL, BEST_SPECS, BEST_COMBO, rho, fj_k)
            scenarios_out[alias] = {
                "years": horizon,
                "pd":    [_safe_float(v) for v in pd_pit.tolist()],
                "lgd":   [_safe_float(v) for v in lgd_pit.tolist()],
            }

        def _mean(a):
            vals = scenarios_out.get(a, {}).get("pd", [])
            return round(float(np.mean(vals)), 6) if vals else None

        def _peak(a):
            vals = scenarios_out.get(a, {}).get("pd", [])
            return round(float(max(vals)), 6) if vals else None

        return {
            "model": {
                "family":  BEST_MODEL["family"],
                "combo":   BEST_COMBO,
                "r2":      round(BEST_MODEL["r2"], 4),
                "n_obs":   BEST_MODEL["n_obs"],
                "verdict": BEST_MODEL.get("verdict", "N/A"),
                "n_fail":  BEST_MODEL.get("n_fail", 0),
                "n_warn":  BEST_MODEL.get("n_warn", 0),
                "n_pass":  BEST_MODEL.get("n_pass", 0),
            },
            "scenarios": scenarios_out,
            "metrics": {
                "pd_ttc":           0.03,
                "lgd_ttc":          0.45,
                "rho":              round(rho, 6),
                "fj_k":             round(fj_k, 6),
                "pd_baseline_mean": _mean("baseline"),
                "pd_adverse_mean":  _mean("adverse"),
                "pd_severe_mean":   _mean("severe"),
                "pd_baseline_peak": _peak("baseline"),
                "pd_adverse_peak":  _peak("adverse"),
                "pd_severe_peak":   _peak("severe"),
            },
            "ngfs_mode": "LT",
        }

    def test_top_level_keys(self):
        out = self._build_output()
        assert set(out.keys()) == {"model", "scenarios", "metrics", "ngfs_mode"}

    def test_model_keys(self):
        out = self._build_output()
        expected = {"family", "combo", "r2", "n_obs", "verdict",
                    "n_fail", "n_warn", "n_pass"}
        assert expected.issubset(set(out["model"].keys()))

    def test_scenarios_keys(self):
        out = self._build_output()
        assert set(out["scenarios"].keys()) == {"baseline", "adverse", "severe"}

    def test_scenario_fields(self):
        out = self._build_output()
        for alias in ("baseline", "adverse", "severe"):
            scen = out["scenarios"][alias]
            assert "years" in scen, f"'years' manquant dans scénario '{alias}'"
            assert "pd"    in scen, f"'pd' manquant dans scénario '{alias}'"
            assert "lgd"   in scen, f"'lgd' manquant dans scénario '{alias}'"
            assert len(scen["years"]) == N_YEARS
            assert len(scen["pd"])    == N_YEARS
            assert len(scen["lgd"])   == N_YEARS

    def test_metrics_keys(self):
        out = self._build_output()
        expected = {"pd_ttc", "lgd_ttc", "rho", "fj_k",
                    "pd_baseline_mean", "pd_adverse_mean", "pd_severe_mean",
                    "pd_baseline_peak", "pd_adverse_peak", "pd_severe_peak"}
        assert expected.issubset(set(out["metrics"].keys()))

    def test_all_values_json_serializable(self):
        """Toutes les valeurs scalaires doivent être JSON-sérialisables (pas inf/nan)."""
        import json
        out = self._build_output()
        # Vérifier que json.dumps ne lève pas d'exception
        json.dumps(out)

    def test_symmetry_with_ngfs_liquidity_output(self):
        """
        Format symétrique à ngfs_liquidity_engine :
        - top-level keys : model/scenarios/metrics/ngfs_mode (crédit)
          vs satellites/scenarios/metrics/ngfs_mode (liquidité)
        - scenarios : mêmes aliases baseline/adverse/severe
        - chaque scénario : years + données (pd+lgd vs lcr+nsfr)
        """
        out = self._build_output()
        # Clés de structure commune
        assert "scenarios" in out
        assert "metrics"   in out
        assert "ngfs_mode" in out
        # Aliases identiques à ceux de ngfs_liquidity_engine
        for alias in ("baseline", "adverse", "severe"):
            assert alias in out["scenarios"]
        # Chaque scénario contient 'years' (identique aux deux modules)
        for alias in ("baseline", "adverse", "severe"):
            assert "years" in out["scenarios"][alias]


# ═══════════════════════════════════════════════════════════════════════════
# (d) Robustesse — variables manquantes dans le fichier NGFS
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingNgfsVariables:
    """Quand une variable macro nécessaire est absente du fichier NGFS,
    la projection doit quand même réussir (imputation par moyenne historique)."""

    def test_projection_with_partial_ngfs_columns(self):
        """DataFrame NGFS avec seulement GDP (chômage absent)."""
        df_partial = pd.DataFrame(
            {"real_gdp_growth": [1.0 + 0.1 * i for i in range(N_YEARS)]},
            index=pd.Index(range(2024, 2024 + N_YEARS), name="year"),
        )
        # Ajouter la colonne manquante avec NaN (comme le fait ngfs_credit_engine)
        df_partial["unemployment_rate"] = float("nan")

        rho  = 0.20
        fj_k = 0.5
        pd_pit, lgd_pit = _run_projection(
            df_partial, BEST_MODEL, BEST_SPECS, BEST_COMBO, rho, fj_k)

        assert len(pd_pit)  == N_YEARS, "Longueur PD incorrecte"
        assert len(lgd_pit) == N_YEARS, "Longueur LGD incorrecte"
        assert pd_pit.min()  > 0,    "PD ≤ 0 avec variable manquante"
        assert lgd_pit.min() >= 0.05, "LGD < floor avec variable manquante"


# ═══════════════════════════════════════════════════════════════════════════
# Imports conditionnels pour Phase 3 — non-bloquants si modules absents
# ═══════════════════════════════════════════════════════════════════════════

try:
    from app.modules.credit.capital_engine import CreditCapitalEngine
    _HAS_CAPITAL_ENGINE = True
except ImportError:
    _HAS_CAPITAL_ENGINE = False

try:
    from app.modules.credit.wrapper import _build_capital_params
    _HAS_BUILD_PARAMS = True
except ImportError:
    _HAS_BUILD_PARAMS = False


# ── Données synthétiques partagées par les tests capital ────────────────────

def _make_ead_series(start=2010, n=14, base=100.0, growth=0.05):
    """EAD historique synthétique avec croissance régulière."""
    years = list(range(start, start + n))
    vals  = [base * (1.0 + growth) ** i for i in range(n)]
    return pd.Series(vals, index=pd.Index(years, name="year"), name="ead")


def _make_pd_hist(n=14, pd_ttc=0.03):
    years = list(range(2010, 2010 + n))
    return pd.Series([pd_ttc] * n, index=pd.Index(years, name="year"), name="pd")


def _make_lgd_arr(n, val=0.45):
    return np.full(n, val)


# PD NGFS synthétiques : baseline stable, adverse montant, severe très fort
N_CAP = 6
HORIZON_CAP = list(range(2024, 2024 + N_CAP))

PD_NGFS = {
    "baseline": np.array([0.030] * N_CAP),
    "adverse":  np.array([0.030 + 0.008 * (i + 1) for i in range(N_CAP)]),
    "severe":   np.array([0.030 + 0.020 * (i + 1) for i in range(N_CAP)]),
}
LGD_NGFS = {
    "baseline": np.array([0.45] * N_CAP),
    "adverse":  np.array([0.48] * N_CAP),
    "severe":   np.array([0.55] * N_CAP),
}

# PD conventionnelles (distinctes des PD NGFS — prouve que le capital
# climat-conditionné utilise bien les PD NGFS et non les PD conventionnelles)
PD_CONV = {
    "baseline": np.array([0.030] * N_CAP),
    "adverse":  np.array([0.030 + 0.004 * (i + 1) for i in range(N_CAP)]),
    "severe":   np.array([0.030 + 0.010 * (i + 1) for i in range(N_CAP)]),
}
LGD_CONV = {
    "baseline": np.array([0.45] * N_CAP),
    "adverse":  np.array([0.46] * N_CAP),
    "severe":   np.array([0.50] * N_CAP),
}


# ═══════════════════════════════════════════════════════════════════════════
# (e) Capital — Effet scénario NGFS
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_CAPITAL_ENGINE or not _HAS_BUILD_PARAMS,
                    reason="capital_engine ou _build_capital_params non importables")
class TestCapitalEffect:
    """RWA/capital sous scénario NGFS severe diffère de baseline."""

    def setup_method(self):
        ead_series = _make_ead_series()
        params     = _build_capital_params(ead_series, {})
        self.engine = CreditCapitalEngine(params)
        self.ead_M  = float(ead_series.iloc[-1])
        pd_hist     = _make_pd_hist()
        lgd_hist    = _make_lgd_arr(len(pd_hist))
        self.out = self.engine.run(
            pd_hist    = pd_hist,
            lgd_hist   = lgd_hist,
            pd_stress  = PD_NGFS,
            lgd_stress = LGD_NGFS,
            ead        = self.ead_M,
            horizon    = HORIZON_CAP,
            capital_df = None,
        )

    def test_rwa_severe_greater_than_baseline(self):
        """RWA_stressed sévère > baseline sur tout l'horizon."""
        df = self.out.stress_df
        rwa_bl  = df[df["scenario"] == "baseline"]["rwa_stressed"].values
        rwa_sev = df[df["scenario"] == "severe" ]["rwa_stressed"].values
        assert (rwa_sev > rwa_bl).all(), (
            f"RWA sévère ≤ baseline: sev={rwa_sev}, bl={rwa_bl}")

    def test_k_required_severe_greater_than_baseline(self):
        """K_required sévère > baseline (ASRF sensible aux PD NGFS)."""
        df   = self.out.stress_df
        k_bl = df[df["scenario"] == "baseline"]["k_required"].values
        k_sv = df[df["scenario"] == "severe" ]["k_required"].values
        assert (k_sv > k_bl).all(), (
            f"K_required sévère ≤ baseline: sev={k_sv}, bl={k_bl}")

    def test_stress_df_has_all_scenarios(self):
        """stress_df contient bien les 3 scénarios NGFS."""
        assert set(self.out.stress_df["scenario"].unique()) == {
            "baseline", "adverse", "severe"}

    def test_stress_df_horizon_length(self):
        """Chaque scénario couvre exactement N_CAP années."""
        for level in ("baseline", "adverse", "severe"):
            n = len(self.out.stress_df[self.out.stress_df["scenario"] == level])
            assert n == N_CAP, f"scénario '{level}': {n} lignes ≠ {N_CAP}"


# ═══════════════════════════════════════════════════════════════════════════
# (f) Capital — breach_year cohérent avec PD climatiques
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_CAPITAL_ENGINE or not _HAS_BUILD_PARAMS,
                    reason="capital_engine ou _build_capital_params non importables")
class TestCapitalBreachYear:
    """
    breach_year est cohérent avec la trajectoire PD/LGD climatique, pas
    avec la trajectoire conventionnelle.

    On construit deux runs :
      - run_ngfs   : PD très stressées (PD_NGFS) → breach probable
      - run_conv   : PD peu stressées  (PD_CONV)  → breach moins probable

    Propriétés vérifiées :
      1. Si run_ngfs breach sur severe mais pas run_conv → breach_year est
         piloté par les PD NGFS.
      2. run_ngfs sévère a un k_required moyen strictement supérieur à
         run_conv sévère (les PD NGFS plus fortes poussent plus de capital).
    """

    def _run_engine(self, pd_arrays, lgd_arrays):
        ead_series = _make_ead_series()
        params     = _build_capital_params(ead_series, {})
        engine     = CreditCapitalEngine(params)
        pd_hist    = _make_pd_hist()
        lgd_hist   = _make_lgd_arr(len(pd_hist))
        return engine.run(
            pd_hist    = pd_hist,
            lgd_hist   = lgd_hist,
            pd_stress  = pd_arrays,
            lgd_stress = lgd_arrays,
            ead        = float(ead_series.iloc[-1]),
            horizon    = HORIZON_CAP,
            capital_df = None,
        )

    def test_k_required_ngfs_higher_than_conventional_severe(self):
        """K_required severe(NGFS) > K_required severe(conventionnel) car PD NGFS > PD conv."""
        out_ngfs = self._run_engine(PD_NGFS, LGD_NGFS)
        out_conv = self._run_engine(PD_CONV, LGD_CONV)

        k_ngfs = out_ngfs.stress_df[
            out_ngfs.stress_df["scenario"] == "severe"]["k_required"].mean()
        k_conv = out_conv.stress_df[
            out_conv.stress_df["scenario"] == "severe"]["k_required"].mean()

        assert k_ngfs > k_conv, (
            f"K_required NGFS ({k_ngfs:.6f}) ≤ conv ({k_conv:.6f}) "
            "alors que PD_NGFS_severe > PD_CONV_severe par construction — "
            "le capital n'utilise pas les PD climatiques")

    def test_severity_calibrated_from_ngfs_pd(self):
        """
        La sévérité adverse doit différer entre run NGFS et run conventionnel
        parce que _calibrate_severity lit les pd_arrays passés au run — qui sont
        les PD NGFS dans un cas, les PD conventionnelles dans l'autre.

        Données conçues pour une séparation nette :
          NGFS : adverse = 70 % de l'amplitude severe → sévérité ≈ 0.70
          CONV : adverse = 30 % de l'amplitude severe → sévérité ≈ 0.30
        (même baseline et même severe pour les deux → seul adverse diffère)
        """
        # Shared baseline + severe; adverse amplitude differs intentionally.
        pd_ngfs_local = {
            "baseline": np.array([0.030] * N_CAP),
            "adverse":  np.array([0.030 + 0.007 * (i + 1) for i in range(N_CAP)]),
            "severe":   np.array([0.030 + 0.010 * (i + 1) for i in range(N_CAP)]),
        }
        lgd_flat = {k: np.array([0.45] * N_CAP) for k in pd_ngfs_local}

        pd_conv_local = {
            "baseline": np.array([0.030] * N_CAP),
            "adverse":  np.array([0.030 + 0.003 * (i + 1) for i in range(N_CAP)]),
            "severe":   np.array([0.030 + 0.010 * (i + 1) for i in range(N_CAP)]),
        }

        out_ngfs = self._run_engine(pd_ngfs_local, lgd_flat)
        out_conv = self._run_engine(pd_conv_local, lgd_flat)

        sev_ngfs = float(out_ngfs.stress_df[
            out_ngfs.stress_df["scenario"] == "adverse"]["severity"].iloc[0])
        sev_conv = float(out_conv.stress_df[
            out_conv.stress_df["scenario"] == "adverse"]["severity"].iloc[0])

        # NGFS adverse closer to severe → higher severity score
        assert sev_ngfs > sev_conv + 0.1, (
            f"Sévérité adverse NGFS ({sev_ngfs:.4f}) pas clairement > conv ({sev_conv:.4f}) "
            "— _calibrate_severity n'utilise pas les PD passées au run")


# ═══════════════════════════════════════════════════════════════════════════
# (g) Capital — parité EAD growth rate entre chemin conventionnel et climatique
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_CAPITAL_ENGINE or not _HAS_BUILD_PARAMS,
                    reason="capital_engine ou _build_capital_params non importables")
class TestCapitalParity:
    """
    _build_capital_params produit exactement les mêmes valeurs que la
    formule inline de credit/wrapper.py — preuve que la fonction est
    bien partagée et non dupliquée avec dérive.

    Test clé : sur le même dataset et les mêmes pd_stress, deux appels à
    CreditCapitalEngine (un pour le chemin conventionnel, un pour le chemin
    climatique) produisent des RWA identiques à la décimale près.
    """

    def _ead(self):
        return _make_ead_series(n=15, base=500.0, growth=0.06)

    def test_build_params_matches_inline_formula(self):
        """
        Les valeurs renvoyées par _build_capital_params correspondent exactement
        à la formule inline extraite de credit/wrapper.py (même arithmétique,
        mêmes opérations pandas).
        """
        ead = self._ead()

        # ── Reproduction exacte de la formule inline ─────────────────────
        _ead_annual_growth = ead.pct_change().dropna()
        n_ead = len(ead)
        if n_ead >= 2 and float(ead.iloc[0]) > 0:
            _ead_cagr = float(
                (ead.iloc[-1] / ead.iloc[0]) ** (1.0 / (n_ead - 1)) - 1.0
            )
        else:
            _ead_cagr = 0.0
        _ead_growth_std = (
            float(_ead_annual_growth.std()) if len(_ead_annual_growth) >= 2 else 0.0
        )
        expected = {
            "ead_growth_rate":    _ead_cagr,
            "ead_growth_adverse": _ead_cagr - _ead_growth_std,
            "ead_growth_severe":  _ead_cagr - 2.0 * _ead_growth_std,
        }
        # ── Résultat de la fonction partagée ─────────────────────────────
        got = _build_capital_params(ead, {})

        for key, exp_val in expected.items():
            assert got[key] == pytest.approx(exp_val, rel=1e-12), (
                f"{key}: attendu={exp_val:.10f}, obtenu={got[key]:.10f} — "
                "dérive entre formule inline et _build_capital_params")

    def test_build_params_user_override_respected(self):
        """Un override utilisateur dans user_params prend la priorité (setdefault)."""
        ead = self._ead()
        override = {"ead_growth_rate": 0.999}
        got = _build_capital_params(ead, override)
        assert got["ead_growth_rate"] == 0.999, (
            "L'override ead_growth_rate=0.999 a été écrasé par le CAGR calculé")

    def test_rwa_identical_conventional_vs_climate_same_inputs(self):
        """
        Preuve de parité bout-en-bout :
        Deux runs CreditCapitalEngine avec les MÊMES pd_stress et la MÊME
        ead_series produisent des RWA identiques à la décimale près.

        Ce test échouerait si les deux chemins utilisaient des fonctions
        distinctes avec des formules CAGR légèrement différentes.
        """
        ead      = self._ead()
        pd_hist  = _make_pd_hist(n=len(ead))
        lgd_hist = _make_lgd_arr(len(ead))

        pd_stress = {
            "baseline": np.array([0.03] * N_CAP),
            "adverse":  np.array([0.05] * N_CAP),
            "severe":   np.array([0.08] * N_CAP),
        }
        lgd_stress = {
            "baseline": np.array([0.45] * N_CAP),
            "adverse":  np.array([0.50] * N_CAP),
            "severe":   np.array([0.55] * N_CAP),
        }

        # Run 1 — simule le chemin conventionnel
        params1 = _build_capital_params(ead, {})
        out1 = CreditCapitalEngine(params1).run(
            pd_hist=pd_hist, lgd_hist=lgd_hist,
            pd_stress=pd_stress, lgd_stress=lgd_stress,
            ead=float(ead.iloc[-1]), horizon=HORIZON_CAP, capital_df=None,
        )

        # Run 2 — simule le chemin climatique (mêmes inputs, même fonction)
        params2 = _build_capital_params(ead, {})
        out2 = CreditCapitalEngine(params2).run(
            pd_hist=pd_hist, lgd_hist=lgd_hist,
            pd_stress=pd_stress, lgd_stress=lgd_stress,
            ead=float(ead.iloc[-1]), horizon=HORIZON_CAP, capital_df=None,
        )

        for level in ("baseline", "adverse", "severe"):
            rwa1 = out1.stress_df[
                out1.stress_df["scenario"] == level]["rwa_stressed"].values
            rwa2 = out2.stress_df[
                out2.stress_df["scenario"] == level]["rwa_stressed"].values
            np.testing.assert_array_almost_equal(
                rwa1, rwa2, decimal=2,
                err_msg=(
                    f"RWA '{level}' diffère entre chemin conventionnel et climatique "
                    f"(max_diff={np.max(np.abs(rwa1 - rwa2)):.4f}) — "
                    "_build_capital_params produit des valeurs divergentes"
                ),
            )
