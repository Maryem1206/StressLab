"""FIGURE 3.2.8 - Evolution du CAR sous les trois scenarios (2025-2028).
Source: RESULTATS_CHAPITRE3.md, section Impact prudentiel (run 7e0b2325)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, style_axes, BLUE, ORANGE, RED, GRID

years = [2025, 2026, 2027, 2028]
car_baseline = [0.200132, 0.176924, 0.156408, 0.138270]
car_adverse = [0.200132, 0.178869, 0.152762, 0.129492]
car_severe = [0.200132, 0.180182, 0.147374, 0.120528]

THRESH_MIN = 0.1050
THRESH_BUFFER = 0.1275
DEFICIT_MAX = 1_088_775  # thousands EGP

fig, ax = plt.subplots(figsize=(9, 5.2))

ax.plot(years, car_baseline, color=BLUE, lw=2.25, marker="o", ms=5, label="Baseline")
ax.plot(years, car_adverse, color=ORANGE, lw=2.25, marker="o", ms=5, label="Adverse")
ax.plot(years, car_severe, color=RED, lw=2.25, marker="o", ms=5, label="Severe")

ax.axhline(THRESH_BUFFER, color=RED, lw=1.1, linestyle=(0, (5, 3)))
ax.axhline(THRESH_MIN, color="#7F7F7F", lw=1.1, linestyle=(0, (5, 3)))
ax.text(2025.02, THRESH_BUFFER + 0.003, "Seuil Bâle III (avec buffer) : 12,75 %",
        fontsize=8.5, color=RED, va="bottom")
ax.text(2025.02, THRESH_MIN - 0.003, "Seuil minimum Bâle III : 10,50 %",
        fontsize=8.5, color="#595959", va="top")

# Breach annotation: Severe crosses 12.75% threshold in 2028
deficit_m_egp = f"{DEFICIT_MAX/1000:,.0f}".replace(",", " ")
ax.annotate(
    f"Breach 2028 (Severe)\nCAR = 12,05 % < 12,75 %\nDéficit de capital ≈ {deficit_m_egp} M EGP",
    xy=(2028, car_severe[-1]), xytext=(2026.15, 0.108),
    fontsize=8.5, color=RED,
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
)

ax.set_xticks(years)
ax.set_xlim(2024.85, 2028.5)
ax.set_ylim(0.09, 0.21)
style_axes(ax, pct_left=True)
ax.set_ylabel("CAR (Total Capital / RWA)")
ax.set_title("FIGURE 3.2.8 — Évolution du CAR sous les trois scénarios (2025–2028)",
              fontsize=12, fontweight="bold", loc="left", pad=14)
ax.legend(loc="upper right", ncol=3, fontsize=9.5)

fig.text(0.01, -0.02,
         "Source : run crédit 7e0b2325 (RESULTATS_CHAPITRE3.md). Seuils Bâle III : 10,50 % (minimum) / 12,75 % (avec conservation buffer).\n"
         "Breach identifié uniquement sous le scénario Severe en 2028 (CAR=12,0528 % < 12,75 %) ; déficit max = 1 088 775 milliers EGP.",
         fontsize=8, color="#595959")

fig.tight_layout(rect=[0, 0.03, 1, 1])
out = os.path.join(os.path.dirname(__file__), "fig_3_2_8_car.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
