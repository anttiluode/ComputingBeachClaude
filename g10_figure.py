"""G10 figure: results/g10.png (held-out beds 1010, 1011)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = [json.load(open(f"results/g10/g10_{b}.json")) for b in (1010, 1011)]
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3dd"
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]      # ordinal: rho
ARM = {"expose": "#2a78d6", "teach": "#eb6834", "ideal_full": "#1baf7a", "ideal_diff": "#eda100"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
for a in ax:
    a.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# (a) frozen readout vs number of reads, per amplitude
rhos = list(R[0]["a"]["by_rho"])
chance = np.mean([r["a"]["chance"] for r in R])
k = np.arange(1, 9)
for i, rho in enumerate(rhos):
    y = np.mean([r["a"]["by_rho"][rho]["single"] for r in R], 0)
    ax[0].plot(k, y, color=RAMP[i], lw=2, marker="o", ms=4)
    if float(rho) >= 4:
        ax[0].text(8.15, y[-1], f"ρ {float(rho):g}", color=INK, va="center", fontsize=8)
ax[0].text(8.15, 6.75, "ρ 0.125–2\n(4 lines, flat)", color=INK, va="center", fontsize=8)
ax[0].axhline(chance, color=MUTED, lw=1, ls="--")
ax[0].text(1, chance + 0.4, "chance", color=MUTED, fontsize=8)
ax[0].set_xlim(0.8, 9.6)
ax[0].set_xlabel("number of physical reads")
ax[0].set_ylabel("storm location error, frozen reader (cells)")
ax[0].set_title("A. Reads below threshold are free; loud reads blind the reader", fontsize=9.5, loc="left")

# (b) what is lost after 8 reads at rho 8
labels = ["unpinged\ncontrol", "frozen\nreader", "efference\nbaseline", "recalibrated\nreader",
          "sand\n(read 1)", "sand\n(read 8)"]
vals = []
for r in R:
    x = r["a"]["by_rho"]["8.0"]
    vals.append([x["ctrl_final"], x["single"][-1], x["eff"][-1], x["recal"]["recalK_cal"], x["sand"]["1"], x["sand"]["8"]])
vals = np.array(vals)
pos = np.arange(len(labels))
ax[1].bar(pos, vals.mean(0), width=0.62, color="#3987e5", edgecolor="white", linewidth=2)
for j, bed in enumerate((1010, 1011)):
    ax[1].scatter(pos + (j - 0.5) * 0.18, vals[j], s=16, color=INK, zorder=3, label=f"bed {bed}" if j == 0 else None)
for p, v, top in zip(pos, vals.mean(0), vals.max(0)):
    ax[1].text(p, max(v, top) + 0.9, f"{v:.1f}", ha="center", fontsize=8)
ax[1].axhline(chance, color=MUTED, lw=1, ls="--")
ax[1].set_ylim(0, 50)
ax[1].set_xticks(pos, labels, fontsize=8)
ax[1].set_ylabel("storm location error (cells)")
ax[1].set_title("A, ρ = 8 after 8 reads: the lens moved, the record mostly stayed", fontsize=9.5, loc="left")
ax[1].text(5.4, 40, "bars: mean of beds 1010, 1011\ndots: each bed", fontsize=7.5, color=MUTED, ha="right")
ax[1].text(-0.4, chance + 0.8, "chance", color=MUTED, fontsize=8)

# (c) learning vs keeping the record
for r in R:
    C = r["c"]["by_cond"]
    for cond, v in C.items():
        amp = cond.split("_")[0]
        beta = cond.split("_")[1]
        for arm, col in ARM.items():
            x, y = v[arm]["J_ratio_median"], v[arm]["sand"]
            mk = "o" if amp == "amp0.45" else "s"
            face = col if beta == "beta0.03" else "white"
            ax[2].scatter(x, y, s=46, marker=mk, facecolor=face, edgecolor=col, linewidth=1.6, zorder=3)
none = np.mean([r["c"]["by_cond"]["amp1.0_beta0.03"]["none"]["sand"] for r in R])
ax[2].axhline(none, color=MUTED, lw=1, ls=":")
ax[2].text(0.0105, none - 1.4, "no episode", color=MUTED, fontsize=8)
ax[2].axhline(chance, color=MUTED, lw=1, ls="--")
ax[2].text(0.0105, chance + 0.4, "chance", color=MUTED, fontsize=8)
ax[2].axvline(1.0, color=MUTED, lw=1, ls=":")
ax[2].set_xscale("log")
ax[2].set_xlim(0.01, 5)
ax[2].set_ylim(0, 24)
ax[2].set_xlabel("task loss after the episode, J / J₀ (log; left = learned)")
ax[2].set_ylabel("storm location error from the sand (cells)")
ax[2].set_title("C. Learning vs keeping the storm record", fontsize=9.5, loc="left")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=c, label=a.replace("_", " ")) for a, c in ARM.items()]
h += [Line2D([], [], marker="o", ls="", mfc=MUTED, mec=MUTED, label="forward peak ≈ 2 σc"),
      Line2D([], [], marker="s", ls="", mfc=MUTED, mec=MUTED, label="forward peak ≈ 10 σc"),
      Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, label="hollow: β 0.3 (filled 0.03)")]
ax[2].legend(handles=h, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()
fig.savefig("results/g10.png", dpi=150)
print("wrote results/g10.png")
