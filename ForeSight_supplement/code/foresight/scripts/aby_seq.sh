#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export FMT=gen8randombattle CHUNK=20 OMP_NUM_THREADS=2
.venv/bin/python run_arm.py gbmh_hp_d2 gbm_hist hp 2 ABYSSAL 200 4 > logs/aby_gbmh_hp_d2.log 2>&1
.venv/bin/python run_arm.py gbmn_hp_d2 gbm_nohist hp 2 ABYSSAL 200 4 > logs/aby_gbmn_hp_d2.log 2>&1
echo ABYDONE
