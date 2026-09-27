#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q STAGE1DONE logs/stage1.log 2>/dev/null; do sleep 5; done
P=".venv/bin/python run_arm.py"
# depth curve with trained opponent model + HP leaf, and trained leaf + trained opp model
for D in 1 3; do $P gbmh_hp_d$D gbm_hist hp $D SH 200 4; done
$P gbmh_gbm_d2 gbm_hist gbm 2 SH 200 4
$P gbmh_lr_d2 gbm_hist logreg 2 SH 200 4
# calibration vs depth: calibrated GBM leaf vs sharpened (logit x3) GBM leaf, depths 1-3
for D in 1 2 3; do
  $P heur_gbm_d$D heur gbm $D SH 200 4
  $P heur_gbmx3_d$D heur sharp:3:gbm $D SH 200 4
done
echo CPU2DONE
