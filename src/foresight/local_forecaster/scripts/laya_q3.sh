#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q LAYAQ2_DONE logs/laya_q2.log 2>/dev/null; do sleep 15; done
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15 OMP_NUM_THREADS=2
P=.venv/bin/python
FMT=gen8randombattle CHUNK=20 $P run_arm.py layah8_fp_d2 laya_hist:laya_opp_hist_gen8randombattle.pt fp 2 ABYSSAL 100 4 > logs/arm_layah8_fp_ABY.log 2>&1
$P score_turns.py data/turns_layah8_hp_d2_ABYSSAL_gen8randombattle.pkl > logs/score_turns_hp.log 2>&1
echo LAYAQ3_DONE
