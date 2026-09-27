#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q STAGE1DONE logs/stage1.log 2>/dev/null; do sleep 5; done
P=".venv/bin/python run_arm.py"
# stream A: opponent-model arms (HP leaf fixed)
(
$P heur_hp_d2 heur hp 2 SH 200 4
$P gbmh_hp_d2 gbm_hist hp 2 SH 200 4
$P gbmn_hp_d2 gbm_nohist hp 2 SH 200 4
$P lrh_hp_d2 logreg_hist hp 2 SH 200 4
for O in MBP RND; do
  $P heur_hp_d2 heur hp 2 $O 100 4
  $P gbmh_hp_d2 gbm_hist hp 2 $O 100 4
  $P gbmn_hp_d2 gbm_nohist hp 2 $O 100 4
done
export FMT=gen8randombattle
$P heur_hp_d2 heur hp 2 ABYSSAL 300 4
$P gbmh_hp_d2 gbm_hist hp 2 ABYSSAL 300 4
$P gbmn_hp_d2 gbm_nohist hp 2 ABYSSAL 300 4
) > logs/cpu_streamA.log 2>&1 &
# stream B: leaf arms (heuristic opponent model fixed)
(
$P heur_lr_d2 heur logreg 2 SH 200 4

$P heur_lrdelta_d2 heur logreg_delta 2 SH 200 4
$P heur_hyblr_d2 heur hyb:0.25:logreg 2 SH 200 4
for D in 1 3; do
  $P heur_hp_d$D heur hp $D SH 200 4
  $P heur_lr_d$D heur logreg $D SH 200 4
done
) > logs/cpu_streamB.log 2>&1 &
wait
echo CPU1DONE
