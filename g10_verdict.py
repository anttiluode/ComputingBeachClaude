"""G10 verdict: evaluate the pre-registered pass rules (PREREG_G10.md) on the held-out JSONs.

Usage: python g10_verdict.py results/g10/g10_1010.json results/g10/g10_1011.json > results/g10/verdict.json
Each prediction must hold on every bed given.
"""
import json
import sys


def rules(r):
    A, B, C = r["a"]["by_rho"], r["b"]["by_T"], r["c"]["by_cond"]
    ca, cb = r["a"]["chance"], r["b"]["chance"]
    a8 = A["8.0"]
    out = {}
    out["P10a"] = all(A[k]["wrote_max"] == 0.0 and A[k]["proj_median"][-1] >= 0.9999
                      and abs(A[k]["single"][-1] - A[k]["single"][0]) <= 1.0 for k in ("0.125", "0.5"))
    out["P10b"] = A["2.0"]["proj_median"][-1] >= 0.99 and a8["proj_median"][-1] <= 0.90
    out["P10c"] = a8["single"][-1] >= 1.5 * a8["single"][0] and a8["single"][-1] >= 0.75 * ca
    out["P10d"] = a8["sand"]["8"] <= 1.5 * a8["sand"]["1"] and a8["sand"]["8"] <= 0.5 * ca
    out["P10e"] = a8["eff"][-1] >= 0.75 * ca
    out["P10f"] = "recalK_cal" in a8["recal"] and a8["recal"]["recalK_cal"] <= 0.6 * a8["single"][-1]
    b48, binf = B["48.0"], B["inf"]
    out["P10g"] = b48["ctrl_proj_median"] <= 0.1 and all(
        b48["by_rho"][k]["ctrl_final"] >= 0.9 * cb for k in b48["by_rho"])
    w = binf["by_rho"]["0.125"]
    out["P10h"] = w["proj_median"][-1] >= binf["ctrl_proj_median"] - 0.1 and w["single"][-1] <= w["ctrl_final"] + 3
    s = binf["by_rho"]["2.0"]
    out["P10i"] = s["proj_median"][-1] < binf["ctrl_proj_median"] and s["single"][-1] > s["ctrl_final"]
    c1, c2 = C["amp1.0_beta0.03"], C["amp0.45_beta0.03"]
    out["P10j"] = (c1["expose"]["read_frozen"] >= 1.5 * c1["none"]["read_frozen"]
                   and c1["expose"]["sand"] <= 1.5 * c1["none"]["sand"] and c1["expose"]["proj_median"] >= 0.85)
    out["P10k"] = c1["teach"]["sand"] > c1["expose"]["sand"] and c1["teach"]["proj_median"] < c1["expose"]["proj_median"]
    i = c1["ideal_diff"]
    out["P10l"] = i["J_ratio_median"] <= 0.05 and i["proj_median"] >= 0.9 and i["read_frozen"] <= 1.3 * c1["none"]["read_frozen"]
    t, e = c2["teach"], c2["expose"]
    out["P10m"] = (e["J_ratio_median"] - t["J_ratio_median"] >= 0.05 and c2["teach_beats_expose_frac"] >= 0.6
                   and t["proj_median"] >= 0.95 and t["read_frozen"] <= 1.3 * c2["none"]["read_frozen"])
    out["P10n"] = C["amp0.45_beta0.3"]["teach"]["proj_median"] <= 0.3
    return {k: bool(v) for k, v in out.items()}


if __name__ == "__main__":
    per = {p: rules(json.load(open(p))) for p in sys.argv[1:]}
    keys = next(iter(per.values())).keys()
    v = {k: all(per[p][k] for p in per) for k in keys}
    if v["P10m"]:
        verdict = "yes, in a narrow window: readable while learning when writing stays near threshold"
    elif v["P10l"]:
        verdict = "compatible in principle, but this physical teacher erases at the strength where it learns"
    else:
        verdict = "no: on this bed even ideal learning destroys the record"
    json.dump(dict(per_bed=per, all_beds=v, verdict=verdict), sys.stdout, indent=1)
    print()
