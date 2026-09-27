#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15 OMP_NUM_THREADS=2 FMT=gen8randombattle CHUNK=20
P=".venv/bin/python run_arm.py"
L=laya_opp_hist_gen8randombattle.pt
run() { name=$1; shift; echo "$(date +%H:%M) start $name $*"; "$@" > logs/night_$name.log 2>&1; echo "$(date +%H:%M) end $name rc=$?"; tail -1 data/arm_results.jsonl | cut -c1-260; }
run B1_d1 $P layah8_hp_d1 laya_hist:$L hp 1 ABYSSAL 100 4
run B1_d3 $P layah8_hp_d3 laya_hist:$L hp 3 ABYSSAL 100 2
run B2_d2 $P layah8_hp_d2 laya_hist:$L hp 2 ABYSSAL 100 4
OPP_MASS=0.99 OPP_CAP=5,3 CHANCE=0.97,6,0.9,3 run B3_wide $P layah8_wide_d2 laya_hist:$L hp 2 ABYSSAL 100 2
run B4_x3_d2 $P layah8x3_hp_d2 laya_hist_x3:$L hp 2 ABYSSAL 100 4
run B5_unif $P unif_hp_d2 uniform hp 2 ABYSSAL 100 4
run B4_x3_d1 $P layah8x3_hp_d1 laya_hist_x3:$L hp 1 ABYSSAL 100 4
run B4_x3_d3 $P layah8x3_hp_d3 laya_hist_x3:$L hp 3 ABYSSAL 100 2
echo NIGHT_BATTLES_DONE
