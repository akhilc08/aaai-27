#!/bin/zsh
# Runs the four pilots sequentially, then the analysis. Each experiment is
# resumable: re-running skips trials already in results/*.jsonl.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs results figures
echo "=== start $(date)"
python exp2_ratchet.py            2>&1 | tee -a logs/exp2.log
python exp1_repair_retention.py   2>&1 | tee -a logs/exp1.log
python exp3_ratchet_rescue.py     2>&1 | tee -a logs/exp3.log
python exp4_nll_gap.py            2>&1 | tee -a logs/exp4.log
python analyze.py                 2>&1 | tee -a logs/analyze.log
echo "=== done $(date)"
