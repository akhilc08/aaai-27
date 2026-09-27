#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q STAGE1DONE logs/stage1.log 2>/dev/null; do sleep 5; done
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3
HIST=1 TASKS=opp .venv/bin/python train_laya.py 4 0 0 10000 opp_hist > logs/train_opp_hist.log 2>&1
HIST=0 TASKS=opp NOZS=1 .venv/bin/python train_laya.py 4 0 0 10000 opp_nohist > logs/train_opp_nohist.log 2>&1
echo Q1DONE
