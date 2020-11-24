#!/bin/bash

while getopts a:b:h:t: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        h) hs=${OPTARG};;
        t) ts=${OPTARG};;
    esac
done

for h in $hs
  do
  echo $alpha $beta $h $ts
  python mds/compute_1d_hjb_solution.py --alpha $alpha \
                                        --beta $beta \
                                        --h $h \
    					--target-set $ts
  done
