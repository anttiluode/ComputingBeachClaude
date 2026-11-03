#!/bin/sh
# G10 held-out runs (PREREG_G10.md).  Two cores: one lane per bed.
set -e
mkdir -p results/g10
for bed in 1010 1011; do
  ( python g10_run.py a $bed 400 150 results/g10/a_$bed.npz
    python g10_run.py c $bed 400 150 results/g10/c_$bed.npz
    python g10_run.py b $bed 200 60 results/g10/b_$bed.npz
    python g10_analyse.py results/g10/g10_$bed.json results/g10/a_$bed.npz results/g10/b_$bed.npz results/g10/c_$bed.npz \
      > results/g10/analyse_$bed.txt 2>&1 ) > results/g10/lane_$bed.log 2>&1 &
done
wait
