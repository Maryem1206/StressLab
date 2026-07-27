"""FIGURE 3.3.6 - Trajectoires LCR sous les trois scenarios (2024-2028).
Source: RESULTATS_LIQUIDITE_V2.md section 7 (run 7bcec008 / d606a1b7)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, style_axes, BLUE, ORANGE, RED

years = [2024, 2025, 2026, 2027, 2028]
lcr_baseline = [185.20, 185.65, 191.54, 196.54, 200.62]
lcr_adverse = [185.20, 185.68, 176.46, 191.41, 199.90]
lcr_severe = [185.20, 185.74, 164.51, 187.01, 199.35]

fig, ax = plt.subplots(figsize=(9, 5.2))

ax.plot(years, lcr_baseline, color=BLUE, lw=2.25, marker="o", ms=5, label="Baseline")
ax.plot(years, lcr_adverse, color=ORANGE, lw=2.25, marker="o", ms=5, label="Adverse")
ax.plot(years, lcr_severe, color=RED, lw=2.25, marker="o", ms=5, label="Severe")

ax.axhline(100, color=RED, lw=1.1, linestyle=(0, (5, 3)))
ax.text(2024.02, 103, "Seuil réglementaire LCR : 100 %", fontsize=8.5, color=RED)

min_severe = min(lcr_severe)
min_year = years[lcr_severe.index(min_severe)]
ax.annotate(f"LCR minimum sous Sévère\n{min_severe:.2f} % ({min_year})",
            xy=(min_year, min_severe), xytext=(2026.15, 140),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax.set_xticks(years)
ax.set_xlim(2023.85, 2028.4)
ax.set_ylim(0, 230)
style_axes(ax, pct_left=False)
ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
ax.set_ylabel("LCR (%)")
ax.set_title("FIGURE 3.3.6 — Trajectoires LCR sous les trois scénarios (2024–2028)",
              fontsize=12, fontweight="bold", loc="left", pad=14)
ax.legend(loc="lower right", ncol=3, fontsize=9.5)

fig.text(0.01, -0.02,
         "Source : run liquidité 7bcec008 (RESULTATS_LIQUIDITE_V2.md). Aucun franchissement du seuil de 100 % sur l'horizon 2025–2028, "
         "y compris sous le scénario Sévère (minimum 164,51 % en 2026).",
         fontsize=8, color="#595959")

fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(os.path.dirname(__file__), "fig_3_3_6_lcr.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
