#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
until grep -q ALLDONE logs/gen2.log; do sleep 5; done
P=.venv/bin/python; export LIVEHIST=1 RAWLOG=1 OUTDIR=data_raw
$P gen_data.py SH SH 400 10
$P gen_data.py H2 SH 400 10
$P gen_data.py SH MBP 200 10
$P gen_data.py H2 MBP 100 10
$P gen_data.py SH RND 100 10
INDIR=data_raw $P build_ds.py gen9randombattle
echo RAWDONE
