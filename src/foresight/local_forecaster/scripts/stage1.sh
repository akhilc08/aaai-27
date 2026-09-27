#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q ALLDONE logs/gen2.log; do sleep 5; done
P=.venv/bin/python
$P build_ds.py gen9randombattle > logs/build9.log 2>&1
$P build_ds.py gen8randombattle > logs/build8.log 2>&1
$P eval_base.py gen9randombattle > logs/base9.log 2>&1
$P eval_base.py gen8randombattle > logs/base8.log 2>&1
echo STAGE1DONE
