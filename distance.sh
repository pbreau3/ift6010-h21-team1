#!/bin/bash

mkdir -p log/

for model in model/kenlm/*.bin
do
    log=log/distances_$(basename $model).log
    for file in "data/JSB Chorales/test/text/"*.txt
    do
        python2 distance_python2.py $model "data/JSB Chorales/voc_musicautobot.voc" "data/JSB Chorales/test/text/"$file \
            >> $log
    done
done

stats="from statistics import mean, stdev
import sys

nums = []
for line in sys.stdin:
    nums.append(int(line.strip()))

print(mean(nums), stdev(nums), sep=\\t)"

GREEN='\033[0;32m'
NC='\033[0m'

for file in log/*.log
do
    mean_stdev=$(python -c $stats < $file)
    printf "$mean_stdev ${GREEN}$file${NC}"
done