#!/bin/zsh
# Hard memory cap for the local_forecaster experiment (macOS has no cgroups).
# Every 5 s: sum phys_footprint (includes GPU/MPS + compressed memory) of all experiment
# processes; if the total exceeds CAP_MB, SIGTERM the largest one (SIGKILL after 5 s).
CAP_MB=${CAP_MB:-14336}
LOG=/Users/sickle/Coding/aaai-27/src/foresight/local_forecaster/logs/watchdog.log
PAT='local_forecaster|pokemon_search/\.venv|mlx_lm|mlx_opp'
echo "$(date '+%F %T') watchdog start cap=${CAP_MB}MB pid=$$" >> $LOG
while true; do
  total=0; maxmb=0; maxpid=""
  for p in $(pgrep -f "$PAT"); do
    [ "$p" = "$$" ] && continue
    mb=$(footprint $p 2>/dev/null | awk '/phys_footprint:/ {v=$2; if($3=="GB") v*=1024; if($3=="KB") v/=1024; print int(v); exit}')
    [ -z "$mb" ] && continue
    total=$((total+mb))
    if [ $mb -gt $maxmb ]; then maxmb=$mb; maxpid=$p; fi
  done
  if [ $total -gt $CAP_MB ] && [ -n "$maxpid" ]; then
    echo "$(date '+%F %T') OVER CAP total=${total}MB > ${CAP_MB}MB; killing pid $maxpid (${maxmb}MB): $(ps -o command= -p $maxpid | cut -c1-120)" >> $LOG
    kill -TERM $maxpid 2>/dev/null; sleep 5; kill -KILL $maxpid 2>/dev/null
  fi
  sleep 5
done
