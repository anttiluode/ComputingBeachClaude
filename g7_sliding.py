"""G7: sand with a sliding threshold, one teacher exposure per epoch, no subtraction in
code (PREREG_G6G7.md)."""
import json
import numpy as np

from beach.medium import Beach, rough_bed
from beach.sliding import Task, run_arm

W, BETA, EPOCHS, STEP = 0.40, 0.003, 20, 0.0005
DAMPS = [0.0, 0.0005, 0.006]
SEEDS = range(600, 608)
ARMS = [("gd", np.inf), ("natural", np.inf), ("sliding", np.inf), ("sliding_T3", 3.0),
        ("sliding0", np.inf), ("twophase", np.inf)]
TASKS = {"quiet": 1, "loud": -1}

out = {"config": dict(W=W, BETA=BETA, EPOCHS=EPOCHS, STEP=STEP), "runs": {}}
for damp in DAMPS:
    bch = Beach(damp=damp)
    for tname, sign in TASKS.items():
        key = f"{damp}_{tname}"
        beds = []
        for seed in SEEDS:
            sand = rough_bed(bch, seed)
            task = Task(bch, W, bch.port("west"), bch.port("harbourN"), sign, BETA)
            arms = {}
            for name, T in ARMS:
                arm = "sliding" if name == "sliding_T3" else name
                arms[name] = run_arm(task, sand, arm, EPOCHS, STEP, T)
            J0, Jgd = arms["gd"]["J"][0], arms["gd"]["J"][-1]
            for a in arms.values():
                a["f"] = float((J0 - a["J"][-1]) / (J0 - Jgd)) if J0 != Jgd else 0.0
            beds.append(dict(seed=seed, arms=arms))
            print(key, seed, " ".join(f"{n}:f{a['f']:+.2f}/c1{a['cos'][0]:+.2f}" for n, a in arms.items()), flush=True)
        med = lambda n, k: float(np.median([b["arms"][n][k] for b in beds]))
        summ = dict(
            f={n: med(n, "f") for n, _ in ARMS},
            cos_epoch1={n: float(np.median([b["arms"][n]["cos"][0] for b in beds])) for n, _ in ARMS},
            cos_epoch_last={n: float(np.median([b["arms"][n]["cos"][-1] for b in beds])) for n, _ in ARMS},
            sliding_beats_natural=int(sum(b["arms"]["sliding"]["J"][-1] < b["arms"]["natural"]["J"][-1] for b in beds)),
            sliding0_max_raw=float(max(max(b["arms"]["sliding0"]["raw_max"]) for b in beds)),
        )
        out["runs"][key] = dict(beds=beds, summary=summ)
        print(key, summ, flush=True)

R = out["runs"]
L = [R[f"0.0_{t}"]["summary"] for t in TASKS]
out["P7a_pass"] = bool(all(s["cos_epoch1"]["sliding"] >= 0.95 for s in L))
out["P7b_pass"] = bool(all(s["sliding_beats_natural"] >= 7 for s in L))
out["P7c_pass"] = bool(all(R[k]["summary"]["sliding0_max_raw"] == 0.0 for k in R))
out["P7d_pass"] = bool(all(s["f"]["sliding"] >= 0.5 for s in L))
out["P7e_pass"] = bool(all(s["f"]["sliding_T3"] < s["f"]["sliding"] for s in L))
out["P7f_pass"] = bool(all(s["cos_epoch_last"]["sliding"] < s["cos_epoch1"]["sliding"] - 0.05 for s in L))
out["P7g_pass"] = bool(all(R[f"0.006_{t}"]["summary"]["cos_epoch1"]["sliding"] <= 0.4 for t in TASKS))
json.dump(out, open("results/g7.json", "w"), indent=1)
print({k: v for k, v in out.items() if k.startswith("P7")})
