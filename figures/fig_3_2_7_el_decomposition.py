"""FIGURE 3.2.7 - Expected Loss agregee sous les 3 scenarios (2026-2028),
decomposee en contribution PD et contribution LGD (effet Frye-Jacobs).
Source: RESULTATS_CHAPITRE3.md (run 7e0b2325) - PD et LGD Frye-Jacobs trajectories.
Decomposition exacte: EL_t - EL_t0 = (PD_t-PD_t0)*LGD_t0 + PD_t*(LGD_t-LGD_t0)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, style_axes, BLUE, ORANGE, RED, GRAY

years = [2026, 2027, 2028]
PD_t0, LGD_t0 = 0.040000, 0.423021

PD = {
    "Baseline": [0.044095, 0.043108, 0.041598],
    "Adverse": [0.057424, 0.067197, 0.075405],
    "Severe": [0.074467, 0.103239, 0.132753],
}
LGD = {
    "Baseline": [0.432812, 0.430512, 0.426922],
    "Adverse": [0.460739, 0.478395, 0.491869],
    "Severe": [0.490383, 0.531062, 0.565298],
}
COLORS = {"Baseline": BLUE, "Adverse": ORANGE, "Severe": RED}

EL_t0 = PD_t0 * LGD_t0

fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), sharey=True)

for ax, scen in zip(axes, ["Baseline", "Adverse", "Severe"]):
    pd_c, lgd_c = [], []
    el_totals = []
    for i, y in enumerate(years):
        pd_t, lgd_t = PD[scen][i], LGD[scen][i]
        contrib_pd = (pd_t - PD_t0) * LGD_t0
        contrib_lgd = pd_t * (lgd_t - LGD_t0)
        pd_c.append(contrib_pd)
        lgd_c.append(contrib_lgd)
        el_totals.append(pd_t * lgd_t)

    x = range(len(years))
    base = [EL_t0] * len(years)
    ax.bar(x, base, color="#D9D9D9", label="EL base (2025)", width=0.55)
    ax.bar(x, pd_c, bottom=base, color=COLORS[scen], alpha=0.55, label="Contribution PD", width=0.55)
    bottom2 = [b + p for b, p in zip(base, pd_c)]
    ax.bar(x, lgd_c, bottom=bottom2, color=COLORS[scen], alpha=1.0, label="Contribution LGD (Frye-Jacobs)", width=0.55)

    for xi, tot in zip(x, el_totals):
        ax.annotate(f"{tot*100:.2f}%", (xi, tot), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=8.5, fontweight="bold", color="#262626")

    style_axes(ax, pct_left=True)
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_title(scen, fontsize=11, fontweight="bold", color=COLORS[scen])
    ax.set_ylim(0, 0.085)

axes[0].set_ylabel("Expected Loss (PD × LGD)")
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9.5)

fig.suptitle("FIGURE 3.2.7 — Expected Loss agrégée sous les trois scénarios (2026–2028)",
             fontsize=13, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, -0.11,
         "Source : run crédit 7e0b2325 (RESULTATS_CHAPITRE3.md). EL_t = PD_t × LGD_t. Décomposition exacte de ΔEL vs 2025 : "
         "contribution PD = ΔPD × LGD_2025 ; contribution LGD = PD_t × ΔLGD (effet d'amplification Frye-Jacobs, k=0,229776).",
         fontsize=8, color="#595959")

fig.tight_layout(rect=[0, 0.05, 1, 0.93])
out = os.path.join(os.path.dirname(__file__), "fig_3_2_7_el_decomposition.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
