#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q Q1DONE logs/gpu_q1.log 2>/dev/null; do sleep 10; done
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
P=.venv/bin/python
G=gen9randombattle
$P run_arm.py layah_hp_d2 laya_hist:laya_opp_hist_$G.pt hp 2 SH 150 4 > logs/arm_layah_SH.log 2>&1
FMT=gen8randombattle MODEL_FMT=$G CHUNK=20 $P run_arm.py layah_hp_d2 laya_hist:laya_opp_hist_$G.pt hp 2 ABYSSAL 200 4 > logs/arm_layah_ABY.log 2>&1
$P run_arm.py layah_hp_d1 laya_hist:laya_opp_hist_$G.pt hp 1 SH 150 4 > logs/arm_layah_d1.log 2>&1
$P run_arm.py layah_hp_d3 laya_hist:laya_opp_hist_$G.pt hp 3 SH 150 4 > logs/arm_layah_d3.log 2>&1
$P run_arm.py layan_hp_d2 laya_nohist:laya_opp_nohist_$G.pt hp 2 SH 150 4 > logs/arm_layan_SH.log 2>&1
TASKS=pwin,delta NOZS=1 $P train_laya.py 4 16000 8000 0 leaf > logs/train_leaf.log 2>&1
$P run_arm.py heur_laya_d2 heur laya:laya_leaf_$G.pt 2 SH 100 4 > logs/arm_heur_laya.log 2>&1
$P run_arm.py heur_hyblaya_d2 heur hyb:0.25:laya:laya_leaf_$G.pt 2 SH 100 4 > logs/arm_heur_hyblaya.log 2>&1
$P run_arm.py heur_layadelta_d2 heur layadelta:laya_leaf_$G.pt 2 SH 100 4 > logs/arm_heur_layadelta.log 2>&1
echo Q2DONE
