#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q Q2DONE logs/gpu_q2.log 2>/dev/null; do sleep 10; done
export PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
P=.venv/bin/python
export INDIR=data_raw
# raw-log experiment (same rows): Laya / ModernBERT / Qwen on raw log vs hand-feature text; GBM baselines from eval_base on data_raw
TEXT=raw TASKS=opp ZSTAG=_raw $P train_laya.py 4 0 0 8000 raw_opp > logs/tr_laya_raw_opp.log 2>&1
TEXT=hand HIST=1 TASKS=opp NOZS=1 $P train_laya.py 4 0 0 8000 hand_opp > logs/tr_laya_hand_opp.log 2>&1
TEXT=both HIST=1 TASKS=opp NOZS=1 $P train_laya.py 4 0 0 8000 both_opp > logs/tr_laya_both_opp.log 2>&1
TEXT=raw TASKS=pwin NOZS=1 $P train_laya.py 4 12000 0 0 raw_pwin > logs/tr_laya_raw_pwin.log 2>&1
TEXT=hand TASKS=pwin NOZS=1 $P train_laya.py 4 12000 0 0 hand_pwin > logs/tr_laya_hand_pwin.log 2>&1
TEXT=raw $P train_mbert.py opp_hist 8000 raw_opp > logs/tr_mbert_raw_opp.log 2>&1
TEXT=hand $P train_mbert.py opp_hist 8000 hand_opp > logs/tr_mbert_hand_opp.log 2>&1
TEXT=raw $P train_mbert.py pwin 12000 raw_pwin > logs/tr_mbert_raw_pwin.log 2>&1
TEXT=hand $P train_mbert.py pwin 12000 hand_pwin > logs/tr_mbert_hand_pwin.log 2>&1
TEXT=hand $P train_lm.py opp_hist 0 zs_hand_opp > logs/tr_lm_zs_hand_opp.log 2>&1
TEXT=raw $P train_lm.py opp_hist 4000 lora_raw_opp > logs/tr_lm_lora_raw_opp.log 2>&1
TEXT=hand $P train_lm.py opp_hist 4000 lora_hand_opp > logs/tr_lm_lora_hand_opp.log 2>&1
TEXT=hand $P train_lm.py pwin 0 zs_hand_pwin > logs/tr_lm_zs_hand_pwin.log 2>&1
TEXT=hand $P train_lm.py pwin 6000 lora_hand_pwin > logs/tr_lm_lora_hand_pwin.log 2>&1
echo Q3DONE
