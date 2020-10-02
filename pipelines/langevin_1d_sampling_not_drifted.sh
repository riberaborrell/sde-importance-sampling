#!/bin/bash

while getopts a:b:M: flag
do
    case "${flag}" in
        a) alphas=${OPTARG};;
        b) betas=${OPTARG};;
        M) trajectories=${OPTARG};;
    esac
done

for alpha in $alphas
do
  for beta in $betas
  do
    for M in $trajectories
    do
    echo $alpha $beta $M
    python mds/langevin_1d_sampling_not_drifted.py --alpha $alpha \
                                                   --beta $beta \
                                                   --M $M
    done
  done
done
