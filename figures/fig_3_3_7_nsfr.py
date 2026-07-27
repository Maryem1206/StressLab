"""FIGURE 3.3.7 - Trajectoires NSFR sous les trois scenarios (2024-2028).
Source: RESULTATS_LIQUIDITE_V2.md section 7 (run 7bcec008 / d606a1b7)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, style_axes, BLUE, ORANGE, RED

years = [2024, 2025, 2026, 2027, 2028]
nsfr_baseline = [107.05, 106.55, 108.61, 109.39, 110.89]
nsfr_adverse = [107.05, 104.65, 104.71, 103.63, 103.17]
nsfr_severe = [107.05, 100.94, 97.39, 92.99, 89.30]

fig, ax = plt.subplots(figsize=(9, 5.2))

ax.plot(years, nsfr_baseline, color=BLUE, lw=2.25, marker="o", ms=5, label="Baseline")
ax.plot(years, nsfr_adverse, color=ORANGE, lw=2.25, marker="o", ms=5, label="Adverse")
ax.plot(years, nsfr_severe, color=RED, lw=2.25, marker="o", ms=5, label="Severe")

ax.axhline(100, color=RED, lw=1.1, linestyle=(0, (5, 3)))
ax.text(2024.02, 101.3, "Seuil réglementaire NSFR : 100 %", fontsize=8.5, color=RED)

ax.annotate("Breach 2026 (Severe)\nNSFR = 97,39 % < 100 %",
            xy=(2026, 97.39), xytext=(2024.3, 88),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
ax.annotate("Trajectoire divergente\njusqu'à 89,30 % (2028)",
            xy=(2028, 89.30), xytext=(2026.6, 82),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax.set_xticks(years)
ax.set_xlim(2023.85, 2028.4)
ax.set_ylim(75, 120)
style_axes(ax, pct_left=False)
ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
ax.set_ylabel("NSFR (%)")
ax.set_title("FIGURE 3.3.7 — Trajectoires NSFR sous les trois scénarios (2024–2028)",
              fontsize=12, fontweight="bold", loc="left", pad=14)
ax.legend(loc="upper right", ncol=3, fontsize=9.5)

fig.text(0.01, -0.02,
         "Source : run liquidité 7bcec008 (RESULTATS_LIQUIDITE_V2.md). Breach réglementaire identifié dès 2026 sous le scénario Sévère "
         "(NSFR=97,39 %), avec dégradation continue jusqu'à 89,30 % en 2028 (déficit de 10,70 pp vs seuil bâlois).",
         fontsize=8, color="#595959")

fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(os.path.dirname(__file__), "fig_3_3_7_nsfr.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
