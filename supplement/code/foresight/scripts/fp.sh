#!/bin/bash
# total phys_footprint (MB) of this pilot's processes
tot=0
for p in $(pgrep -f "local_forecaster|run_arm.py|train_laya|train_lm|train_mbert|abyssal_runner|eval_base|build_ds|pokemon-showdown start 8001"); do
  mb=$(footprint $p 2>/dev/null | awk '/phys_footprint:/{v=$2; u=$3; if(u ~ /GB/) v*=1024; if(u ~ /KB/) v/=1024; print int(v); exit}')
  [ -n "$mb" ] && tot=$((tot+mb)) && [ "$1" == "-v" ] && echo "$p $mb $(ps -o command= -p $p | cut -c1-80)"
done
echo "TOTAL_MB $tot"
