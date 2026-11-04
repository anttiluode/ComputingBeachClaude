"""Score PREREG_G11.  Usage: python g11_verdict.py results/g11/across_1300.json results/g11/within.json"""
import sys
import json

a = json.load(open(sys.argv[1]))
w = json.load(open(sys.argv[2]))
lo, hi = a["low"], a["high"]
v = {}
v["P11a_not_dark"] = lo["order_snr_median"] >= 10
v["P11b_dimmer_and_finer"] = (lo["gain_ratio_order_over_common"] <= 0.6
                              and a["radial_k"]["order"] >= 1.3 * a["radial_k"]["common"])
v["P11c_index_scrambles_content_does_not"] = (lo["transplant_cos_median"] <= 0.4
                                              and lo["same_index_other_order_cos_median"] >= 0.6)
v["P11d_bed_calibrated_reader_works"] = sum(x["read"] >= 0.85 for x in w) >= 6
v["P11e_across_bed_reader_fails"] = a["across_bed_decoding"]["low"] <= 0.80
v["P11f_born_order_yes_storm_no"] = lo["born_rel_err_order"] <= 0.5 and lo["born_rel_err_storm"] >= 1.0
v["P11g_finer_waves_scramble_more"] = hi["transplant_cos_median"] < lo["transplant_cos_median"]
worst = max(a["worst_probe_over_sigma_c"], max(x["worst_probe_over_sigma_c"] for x in w))
out = {"predictions": {k: bool(x) for k, x in v.items()},
       "n_pass": int(sum(v.values())), "of": len(v),
       "reads_write_nothing": bool(worst < 1.0), "worst_probe_over_sigma_c": worst,
       "dark_sand_rejected": bool(v["P11a_not_dark"]),
       "scrambled_index_supported": bool(v["P11c_index_scrambles_content_does_not"]
                                         and v["P11d_bed_calibrated_reader_works"]
                                         and v["P11e_across_bed_reader_fails"])}
print(json.dumps(out, indent=1))
