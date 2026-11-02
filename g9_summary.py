"""G9: score PREREG_G9.md's predictions from the per-run JSON files -> results/g9.json"""
import json

R = "results/g9/"
BEDS = (1000, 1001, 1002)


def J(name):
    return json.load(open(R + name))


def main():
    h1 = {s: J(f"a_h1_{s}.json") for s in BEDS}
    m1 = {s: J(f"mfp_h1_{s}.json")["mfp"] for s in BEDS}
    h2, h3, h4 = J("a_h2_1000_nf1.json"), J("a_h3_1001_s0.01.json"), J("a_h4_1002_4tone.json")
    m4 = J("mfp_h4_1002.json")["mfp"]
    big = J("big_1000.json")["curve"]
    c = lambda r, n: r["curve"][str(n)]
    out = {"P": {}}
    P = out["P"]
    P["P9a"] = all(c(h1[s], n)["BEACH5"] <= 0.6 * c(h1[s], n)["WATCH5"] for s in BEDS for n in (400, 800))
    P["P9b"] = sum(c(h1[s], 400)["BEACH5"] < c(h1[s], 400)["DENSE64"] for s in BEDS) >= 2
    best_big = min(min(v[k] for k in ("binned", "binned_mlp", "pooled", "pooled_mlp"))
                   for key, v in big.items() if key.startswith("WATCH5_n16000"))
    out["big_watch5_best_at_16000"] = best_big
    P["P9c"] = best_big > c(h1[1000], 400)["BEACH5"]
    P["P9d"] = all(c(h1[s], 400)["BEACH5_fs"] >= 2 * c(h1[s], 400)["BEACH5_f8"] for s in BEDS)
    P["P9e"] = all(h1[s]["ho"]["BEACH5_ho"] <= 1.25 * h1[s]["ho"]["BEACH5_in"] for s in BEDS)
    P["P9f"] = all(c(h3, n)["BEACH5"] <= 0.6 * c(h3, n)["WATCH5"] for n in (400, 800))
    P["P9g"] = all(m1[s]["exact"]["k5"] <= 1.0 for s in BEDS)
    P["P9h"] = all(m1[s]["flat"]["k5"] >= c(h1[s], 800)["BEACH5"] and
                   m1[s]["eps0.03"]["k5"] <= c(h1[s], 800)["BEACH5"] for s in BEDS)
    P["P9h_flat_part"] = all(m1[s]["flat"]["k5"] >= c(h1[s], 800)["BEACH5"] for s in BEDS)
    P["P9i"] = all(m1[s]["flat"]["k20"] <= 3.0 for s in BEDS)
    P["P9j"] = m4["flat"]["k5"] <= c(h4, 400)["BEACH5"]
    P["P9k"] = (c(h2, 400)["WATCH5"] < c(h1[1000], 400)["WATCH5"] and
                c(h2, 400)["BEACH5"] <= 0.6 * c(h2, 400)["WATCH5"])
    if not P["P9a"]:
        v = "no"
    elif P["P9b"] and not P["P9h_flat_part"]:
        v = "only against straw men"
    elif P["P9b"] and P["P9h"]:
        v = "yes, in a bounded regime"
    else:
        v = "mixed: see the predictions"
    out["verdict"] = v
    out["h1"], out["h2"], out["h3"], out["h4"] = h1, h2, h3, h4
    out["mfp_h1"], out["mfp_h4"], out["big"] = m1, m4, big
    json.dump(out, open("results/g9.json", "w"), indent=1)
    print(json.dumps(P, indent=1), "\nverdict:", v)


if __name__ == "__main__":
    main()
