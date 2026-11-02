"""Figures for the README: results/summary.png and results/beds.png."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beach.medium import Beach, rough_bed

ph = json.load(open("results/posthoc_g2.json"))
g1 = json.load(open("results/g1.json"))["cells"]
g3 = json.load(open("results/g3.json"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) gradient alignment vs damping
d = sorted((float(k), v) for k, v in ph["damping"].items())
x = [max(k, 1e-4) for k, _ in d]
ax[0].semilogx(x, [v["cos_teacher_e"] for _, v in d], "o-", label="teacher amplitude e (registered)")
ax[0].semilogx(x, [v["cos_teacher_conj_e"] for _, v in d], "s--", label="teacher amplitude conj(e) (post hoc)")
ax[0].axvline(0.006, color="k", lw=0.8, ls=":")
ax[0].text(0.0062, 0.85, "this cavity\n(Q ~ 67)", fontsize=8)
ax[0].axhline(0, color="gray", lw=0.5)
ax[0].set_xlabel("damping (leftmost point = lossless)")
ax[0].set_ylabel("cos(sand step, exact descent)")
ax[0].set_title("G2: the sand follows the gradient only at low loss")
ax[0].legend(fontsize=8)

# (b) order memory: sand vs read
s = [0.005, 0.01, 0.02, 0.04]
for w, c in (("scour", "C0"), ("exner", "C1")):
    ax[1].plot(s, [g1[f"{w}_s{v}_closed"]["sand_upper_bound"] for v in s], "o-", color=c, label=f"{w}: decode from sand")
    ax[1].plot(s, [g1[f"{w}_s{v}_closed"]["read_loo_nu0.01"] for v in s], "x--", color=c, label=f"{w}: decode from ping read")
ax[1].axhline(0.5, color="gray", lw=0.5)
ax[1].set_xscale("log")
ax[1].set_xlabel("write strength s")
ax[1].set_ylabel("held-out bed accuracy, AB vs BA")
ax[1].set_title("G1: the sand remembers order; the ping can't read it")
ax[1].legend(fontsize=8)

# (c) router
arms = ["gd", "teacher_scour", "teacher_scour_one", "teacher_exner", "natural"]
init = np.array(g3["final"]["initial"])
gains = [np.array(g3["final"][a]) - init for a in arms]
for i, gv in enumerate(gains):
    ax[2].scatter(np.full(len(gv), i) + np.linspace(-0.12, 0.12, len(gv)), gv, s=18)
    ax[2].hlines(gv.mean(), i - 0.25, i + 0.25, color="k")
ax[2].axhline(0, color="gray", lw=0.5)
ax[2].set_xticks(range(len(arms)))
ax[2].set_xticklabels(["GD\n(exact)", "teacher\nscour", "teacher\nscour 1ph", "teacher\nexner", "natural\n(no teacher)"], fontsize=8)
ax[2].set_ylabel("contrast gain after 300 epochs (dB)")
ax[2].set_title("G3: routing two swells, damping 0.006")
plt.tight_layout()
plt.savefig("results/summary.png", dpi=120)

# beds: initial vs GD-trained vs teacher-trained, seed 202
b = Beach()
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a_, (name, sand) in zip(ax, [("initial bed (seed 202)", rough_bed(b, 202)),
                                 ("after GD (+38.6 dB)", np.load("results/g3_sand_gd_202.npy")),
                                 ("after teacher-scour (-4.8 dB)", np.load("results/g3_sand_teacher_scour_202.npy"))]):
    im = a_.imshow(b.to_grid(sand), cmap="YlOrBr", vmin=0, vmax=0.6)
    a_.set_title(name, fontsize=10)
    a_.axis("off")
plt.colorbar(im, ax=ax, fraction=0.02, label="sand height (depth = 1 - sand)")
plt.savefig("results/beds.png", dpi=120, bbox_inches="tight")
print("ok")
