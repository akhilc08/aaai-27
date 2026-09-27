#!/bin/bash
# usage: pool.sh JOBFILE NPAR
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
export OMP_NUM_THREADS=2
run() {
  line="$1"; envs=""; args=()
  for w in $line; do if [[ "$w" == *=* && ${#args[@]} -eq 0 ]]; then envs="$envs $w"; else args+=("$w"); fi; done
  name="${args[0]}_${args[4]}_$(echo $envs | tr -cd '0-9')"
  env $envs .venv/bin/python run_arm.py "${args[@]}" > logs/arm_${name}.log 2>&1
  echo "done $line"
}
export -f run
grep -v '^\s*$' "$1" | xargs -P "$2" -I{} bash -c 'run "{}"'
echo POOLDONE
