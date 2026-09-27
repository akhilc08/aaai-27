#!/bin/bash
# Kills ONLY alfworld runner processes (matched by cwd) if cumulative spend exceeds $9.40.
D=/Users/sickle/Coding/aaai-27/experiments/alfworld_predict_act
mine() { for p in $(pgrep -f "run.py"); do [ "$(lsof -a -p $p -d cwd -Fn 2>/dev/null | grep ^n | cut -c2-)" = "$D" ] && echo $p; done; }
while [ -n "$(mine)" ]; do
  s=$(python3 -c "import json;print(sum(json.loads(l)['cost'] for l in open('$D/runs/usage.jsonl')))")
  if python3 -c "import sys;sys.exit(0 if $s>9.40 else 1)"; then kill $(mine); echo "GUARD KILLED at \$$s"; exit 0; fi
  sleep 15
done
echo "done; spend \$$s"
