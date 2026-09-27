#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15 OMP_NUM_THREADS=2
FMT=gen8randombattle MODEL_FMT=gen9randombattle CHUNK=20 .venv/bin/python run_arm.py layah_fp_d2 laya_hist:laya_opp_hist_gen9randombattle.pt fp 2 ABYSSAL 100 4 > logs/arm_layah_fp_ABY.log 2>&1
echo FPARM_DONE
