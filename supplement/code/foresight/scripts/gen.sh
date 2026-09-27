#!/bin/bash
cd /Users/sickle/Coding/aaai-27/src/foresight/local_forecaster
P=.venv/bin/python; export LIVEHIST=1
$P gen_data.py SH SH 800 10
$P gen_data.py H2 SH 800 10
$P gen_data.py SH MBP 500 10
$P gen_data.py H2 MBP 300 10
$P gen_data.py H2 H2 300 10
$P gen_data.py MBP RND 150 10
$P gen_data.py SH RND 150 10
$P gen_data.py H2 RND 150 10
FMT=gen8randombattle $P gen_data.py SH SH 500 10
FMT=gen8randombattle $P gen_data.py H2 SH 500 10
echo ALLDONE
