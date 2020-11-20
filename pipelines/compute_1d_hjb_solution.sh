#!/bin/bash

while getopts a:b:h:t: flag
do
    case "${flag}" in
        a) alphas=${OPTARG};;
        b) betas=${OPTARG};;
        h) hs=${OPTARG};;
        t) ts=${OPTARG};;
    esac
done

for alpha in $alphas
do
  for beta in $betas
  do
    for h in $hs
    do
      echo $alpha $beta $h $ts
      python mds/compute_1d_hjb_solution.py --alpha $alpha \
                                            --beta $beta \
                                            --h $h \
    					    --target-set $ts
    done
  done
done
