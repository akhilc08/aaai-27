#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q LAYAMAIN_DONE logs/laya_main.log 2>/dev/null && grep -q FPARM_DONE logs/fp_arm.log 2>/dev/null; do sleep 15; done
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25 OMP_NUM_THREADS=2
P=.venv/bin/python
FMT=gen8randombattle HIST=1 TASKS=opp NOZS=1 $P train_laya.py 4 0 0 10000 opp_hist > logs/train_opp_hist_gen8.log 2>&1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15
FMT=gen8randombattle CHUNK=20 $P run_arm.py layah8_hp_d2 laya_hist:laya_opp_hist_gen8randombattle.pt hp 2 ABYSSAL 100 4 > logs/arm_layah8_ABY.log 2>&1
$P scripts/bench_laya.py > logs/bench_laya.log 2>&1
$P score_turns.py data/turns_layah_fp_d2_ABYSSAL_gen8randombattle.pkl > logs/score_turns_fp.log 2>&1
$P score_turns.py data/turns_layah8_hp_d2_ABYSSAL_gen8randombattle.pkl > logs/score_turns_hp8.log 2>&1
echo LAYAQ2_DONE
