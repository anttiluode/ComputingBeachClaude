#!/bin/sh
# G9 held-out runs (PREREG_G9.md).  Two cores: run in two lanes.
set -e
mkdir -p results/g9
( python g9_storms.py 1000 1000 1 results/g9/h1_1000.npz
  python g9_storms.py 1002 1000 1 results/g9/h1_1002.npz
  python g9_storms.py 1001 1000 1 results/g9/h3_1001_s0.01.npz 16 0.01
  python g9_watchbig.py gen 1000 17000 results/g9/h5_1000_big.npz ) > results/g9/lane1.log 2>&1 &
( python g9_storms.py 1001 1000 1 results/g9/h1_1001.npz
  python g9_storms.py 1000 1000 1 results/g9/h2_1000_nf1.npz 1
  python g9_storms.py 1002 600 4 results/g9/h4_1002_4tone.npz ) > results/g9/lane2.log 2>&1 &
wait
