"""FIGURE 3.3.5 - Projections des cinq satellites de liquidite (2024-2028).
Source: RESULTATS_LIQUIDITE_V2.md section 6 (run 7bcec008)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, style_axes, BLUE, ORANGE, RED, GRAY

years = [2024, 2025, 2026, 2027, 2028]

data = {
    "run_off_retail": {
        "baseline": [0.1400, 0.14660, 0.14515, 0.14547, 0.14540],
        "adverse":  [0.1400, 0.14660, 0.17344, 0.15515, 0.14694],
        "severe":   [0.1400, 0.14660, 0.20004, 0.16413, 0.14842],
        "fan": True,
    },
    "run_off_corporate": {
        "baseline": [0.3800, 0.38424, 0.38398, 0.38400, 0.38400],
        "adverse":  [0.3800, 0.38424, 0.39442, 0.38632, 0.38429],
        "severe":   [0.3800, 0.38424, 0.40361, 0.38842, 0.38458],
        "fan": True,
    },
    "asf_factor_corporate": {
        "baseline": [0.4000, 0.394572, 0.395063, 0.395019, 0.395023],
        "adverse":  [0.4000, 0.394572, 0.394161, 0.394906, 0.395014],
        "severe":   [0.4000, 0.394572, 0.393425, 0.394808, 0.395006],
        "fan": True,
    },
    "haircut_add": {
        "baseline": [0.1000, 0.07381, 0.04195, 0.01144, 0.00000],
        "fan": False,
    },
    "asf_factor_retail": {
        "baseline": [0.8000, 0.804371, 0.839688, 0.857790, 0.885468],
        "fan": False,
    },
}

titles = {
    "run_off_retail": "Run-off retail",
    "run_off_corporate": "Run-off corporate",
    "asf_factor_corporate": "ASF factor corporate",
    "haircut_add": "Haircut additionnel (HQLA)",
    "asf_factor_retail": "ASF factor retail",
}

fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
axes = axes.flatten()
order = ["run_off_retail", "run_off_corporate", "asf_factor_corporate", "haircut_add", "asf_factor_retail"]

for i, key in enumerate(order):
    ax = axes[i]
    d = data[key]
    if d["fan"]:
        ax.plot(years, d["baseline"], color=BLUE, lw=2, marker="o", ms=4, label="Baseline")
        ax.plot(years, d["adverse"], color=ORANGE, lw=2, marker="o", ms=4, label="Adverse")
        ax.plot(years, d["severe"], color=RED, lw=2, marker="o", ms=4, label="Severe")
        ax.fill_between(years, d["baseline"], d["severe"], color=RED, alpha=0.06)
    else:
        ax.plot(years, d["baseline"], color=GRAY, lw=2.25, marker="o", ms=4.5, label="Baseline = Adverse = Severe")
        ax.text(0.5, 0.06, "Insensibilité scénaristique\n(trajectoire identique sous les 3 scénarios)",
                transform=ax.transAxes, ha="center", fontsize=7.5, color="#595959")
    style_axes(ax, pct_left=True)
    vals = d["baseline"] + (d.get("adverse", []) + d.get("severe", []) if d["fan"] else [])
    if max(vals) - min(vals) < 0.03:
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(titles[key], fontsize=10.5, loc="left")
    ax.set_xticks(years)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best")

axes[-1].axis("off")

fig.suptitle("FIGURE 3.3.5 — Projections des cinq satellites de liquidité (2024–2028)",
             fontsize=13, fontweight="bold", x=0.02, ha="left")
fig.text(0.02, 0.005,
         "Source : run liquidité 7bcec008 (RESULTATS_LIQUIDITE_V2.md, section 6). run_off_retail, run_off_corporate et asf_factor_corporate "
         "divergent selon le scénario (fan chart) ; haircut_add et asf_factor_retail suivent une trajectoire unique, insensible au scénario "
         "sur l'horizon 2025–2028.",
         fontsize=8, color="#595959")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
out = os.path.join(os.path.dirname(__file__), "fig_3_3_5_satellites.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
