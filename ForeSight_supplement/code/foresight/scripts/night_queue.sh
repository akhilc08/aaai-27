#!/bin/bash
# Overnight batch: strictly sequential, one job at a time.
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 FMT=gen8randombattle CHUNK=20
PY=.venv/bin/python
L=laya_opp_hist_gen8randombattle.pt
Q=logs/night_queue.log
baseline() {  # wait until no job process is alive and our footprint is back to baseline
  for i in $(seq 1 120); do
    busy=$(pgrep -f "run_arm.py|train_laya.py|score_turns.py|abyssal_runner.py|pc_pair_runner.py|bench_laya.py" | wc -l | tr -d ' ')
    tot=$(./scripts/fp.sh | awk '/TOTAL_MB/{print $2}')
    if [ "$busy" = "0" ] && [ "${tot:-0}" -lt 3000 ]; then return 0; fi
    sleep 5
  done
  echo "$(date +%H:%M) WARN baseline not reached (busy=$busy tot=$tot)" >> $Q
}
summ() { $PY night_summary.py >> logs/night_summary_err.log 2>&1; }
battle() {  # battle ITEM NAME OPP LEAF DEPTH N CONC [extra env...]
  item=$1; name=$2; opp=$3; leaf=$4; d=$5; n=$6; c=$7; shift 7
  baseline; echo "$(date +%H:%M) start $item $name n=$n $*" >> $Q
  env "$@" PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 $PY run_arm.py $name $opp $leaf $d ABYSSAL $n $c > logs/night_${item}_${name}.log 2>&1
  echo "$(date +%H:%M) end $item $name rc=$? :: $(grep "\"arm\": \"$name\"" data/arm_results.jsonl | tail -1 | cut -c1-200)" >> $Q; summ
}
train() {  # train ITEM TAG K NP NO [extra env...]
  item=$1; tag=$2; k=$3; np_=$4; no=$5; shift 5
  baseline; echo "$(date +%H:%M) start $item train $tag $*" >> $Q
  env NOZS=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.35 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 BS=4 "$@" $PY train_laya.py $k $np_ 0 $no $tag > logs/night_G_${tag}.log 2>&1
  rc=$?; echo "$(date +%H:%M) end $item train $tag rc=$rc $(grep -E '^done|temps|Error|memguard' logs/night_G_${tag}.log | tail -2 | tr '\n' ' ')" >> $Q
  return $rc
}
score() {  # score TAG
  baseline
  [ -f ckpt/laya_$1_gen8randombattle.pt ] || { echo "$(date +%H:%M) skip score $1 (no ckpt)" >> $Q; return; }
  CKPTS=laya_$1_gen8randombattle.pt SCORE_TAG=_$1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 \
    $PY score_turns.py data/turns_abyssal_eval_gen8.pkl > logs/night_score_$1.log 2>&1
  echo "$(date +%H:%M) scored $1 :: $(grep laya logs/night_score_$1.log | grep acc | tail -1 | cut -c1-160)" >> $Q; summ
}
echo "$(date +%H:%M) queue start" >> $Q; summ
# 1. B2: +100 headline games (pooled with the day run's 100)
battle B2 layah8_hp_d2 laya_hist:$L hp 2 100 4
# 2. B1: depth 1 and depth 3
battle B1 layah8_hp_d1 laya_hist:$L hp 1 100 4
battle B1 layah8_hp_d3 laya_hist:$L hp 3 100 2
# 3. G1: data scaling
for n in 1000 3000 30000; do
  t=opp_hist_n$((n/1000))k
  train G1 $t 4 0 $n FMT=gen8randombattle HIST=1 TASKS=opp || train G1 $t 4 0 $n FMT=gen8randombattle HIST=1 TASKS=opp BS=2
  score $t
done
# 4. G2: no history
train G2 opp_nohist 4 0 10000 FMT=gen8randombattle HIST=0 TASKS=opp || train G2 opp_nohist 4 0 10000 FMT=gen8randombattle HIST=0 TASKS=opp BS=2
score opp_nohist
# 5. B6: PokeChamp's OneStepPlayer vs Abyssal (protocol check vs published 44%)
for c in 1 2 3 4 5; do
  baseline; echo "$(date +%H:%M) start B6 onestep chunk $c" >> $Q
  (cd pokechamp && ../../pokemon_search/.venv-pc/bin/python ../pc_pair_runner.py 20 onestep_check > ../logs/night_B6_onestep_$c.log 2>&1)
  echo "$(date +%H:%M) end B6 chunk $c :: $(tail -1 logs/night_B6_onestep_$c.log | cut -c1-160)" >> $Q
done; summ
# 6. B5: uniform opponent floor
battle B5 unif_hp_d2 uniform hp 2 100 4
# 7. B4: overconfident Laya
battle B4 layah8x3_hp_d2 laya_hist_x3:$L hp 2 100 4
battle B4 layah8x3_hp_d1 laya_hist_x3:$L hp 1 100 4
battle B4 layah8x3_hp_d3 laya_hist_x3:$L hp 3 100 2
# 8. B3: wider search
battle B3 layah8_wide_d2 laya_hist:$L hp 2 100 2 OPP_MASS=0.99 OPP_CAP=4,3 CHANCE=0.95,5,0.85,3
# 9. L1: alive-bonus leaf
battle L1 layah8_alive_d2 laya_hist:$L alive0.3 2 100 4
# 10. L2: learned P(win) leaf (20-game smoke, stop if broken)
train L2 pwin8 4 10000 0 FMT=gen8randombattle TASKS=pwin || train L2 pwin8 4 10000 0 FMT=gen8randombattle TASKS=pwin BS=2
summ
if [ -f ckpt/laya_pwin8_gen8randombattle.pt ]; then
  battle L2 layah8_pwinleaf_d2 laya_hist:$L laya:laya_pwin8_gen8randombattle.pt 2 20 4
  w=$(grep '"arm": "layah8_pwinleaf_d2"' data/arm_results.jsonl | tail -1 | $PY -c "import json,sys; x=json.loads(sys.stdin.read()); print(x['wins'])" 2>/dev/null)
  if [ "${w:-0}" -ge 5 ]; then battle L2 layah8_pwinleaf_d2 laya_hist:$L laya:laya_pwin8_gen8randombattle.pt 2 80 4
  else echo "$(date +%H:%M) L2 smoke looked broken (wins=$w/20); stopped" >> $Q; fi
fi
# 11. G3: laya-multilingual + latency
train G3 ml_opp_hist 4 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp BASE_MODEL=convaiinnovations/laya/multilingual || \
  train G3 ml_opp_hist 4 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp BASE_MODEL=convaiinnovations/laya/multilingual BS=2
score ml_opp_hist
baseline; PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 $PY scripts/bench_laya.py laya_opp_hist_gen8randombattle.pt > logs/night_bench_421.log 2>&1
baseline; [ -f ckpt/laya_ml_opp_hist_gen8randombattle.pt ] && PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 $PY scripts/bench_laya.py laya_ml_opp_hist_gen8randombattle.pt > logs/night_bench_ml.log 2>&1
echo "$(date +%H:%M) benches done" >> $Q; summ
# 12. G4: seed
train G4 opp_hist_seed1 4 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp SEED=1 || train G4 opp_hist_seed1 4 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp SEED=1 BS=2
score opp_hist_seed1
# 13. G5: full fine-tune (skip after two failures)
train G5 full_opp_hist -1 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp BS=2 MEMLIMIT_MB=12000 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 || \
  train G5 full_opp_hist -1 0 10000 FMT=gen8randombattle HIST=1 TASKS=opp BS=1 MEMLIMIT_MB=12000 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.45 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4 || \
  echo "$(date +%H:%M) G5 skipped after two failures" >> $Q
score full_opp_hist
echo "$(date +%H:%M) NIGHT_QUEUE_DONE" >> $Q; summ
