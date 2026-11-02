#!/bin/sh
# G9 analyses; each waits for its data file from run_g9.sh.
R=results/g9
w() { until [ -f "$1" ]; do sleep 20; done; sleep 5; }
for s in 1000 1001 1002; do
  w $R/h1_$s.npz; python g9_analyse.py $R/a_h1_$s.json $R/h1_$s.npz 2>> $R/analyse.log
  python g9_mfp.py $R/h1_$s.npz > $R/mfp_h1_$s.json 2>> $R/analyse.log
done
w $R/h2_1000_nf1.npz; python g9_analyse.py $R/a_h2_1000_nf1.json $R/h2_1000_nf1.npz 2>> $R/analyse.log
w $R/h3_1001_s0.01.npz; python g9_analyse.py $R/a_h3_1001_s0.01.json $R/h3_1001_s0.01.npz 2>> $R/analyse.log
w $R/h4_1002_4tone.npz; python g9_analyse.py $R/a_h4_1002_4tone.json $R/h4_1002_4tone.npz 2>> $R/analyse.log
python g9_mfp.py $R/h4_1002_4tone.npz > $R/mfp_h4_1002.json 2>> $R/analyse.log
w $R/h5_1000_big.npz; python -W ignore g9_watchbig.py fit $R/h5_1000_big.npz > $R/big_1000.json 2>> $R/analyse.log
python g9_summary.py
