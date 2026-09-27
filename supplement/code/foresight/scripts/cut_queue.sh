#!/bin/bash
# Stop night_queue.sh right after it launches B4 at depth 2 (the last item the user kept); the running arm continues.
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q "start B4 layah8x3_hp_d2" logs/night_queue.log; do sleep 2; done
pkill -f scripts/night_queue.sh
echo "$(date +%H:%M) queue cut after launching B4 depth 2 (B3, L1, L2, B4 d1/d3, G3, G4, G5 cancelled by user)" >> logs/night_queue.log
