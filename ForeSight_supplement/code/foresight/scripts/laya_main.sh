#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15 OMP_NUM_THREADS=2
P=.venv/bin/python
G=gen9randombattle
FMT=gen8randombattle MODEL_FMT=$G CHUNK=20 $P run_arm.py layah_hp_d2 laya_hist:laya_opp_hist_$G.pt hp 2 ABYSSAL 200 4 > logs/arm_layah_ABY.log 2>&1
$P run_arm.py layah_hp_d2 laya_hist:laya_opp_hist_$G.pt hp 2 SH 200 4 > logs/arm_layah_SH.log 2>&1
echo LAYAMAIN_DONE
