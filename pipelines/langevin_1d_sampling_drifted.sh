#!/bin/bash

while getopts a:b:m:s:t:l:M: flag
do
    case "${flag}" in
        a) alphas=${OPTARG};;
        b) betas=${OPTARG};;
        t) theta_init=${OPTARG};;
        m) m=${OPTARG};;
        s) sigma=${OPTARG};;
        M) M=${OPTARG};;
    esac
done

for alpha in $alphas
do
  for beta in $betas
  do
  echo $alpha $beta $theta_init $m $sigma $M
  python mds/langevin_1d_sampling_drifted.py --alpha $alpha \
                                             --beta $beta \
                                             --theta-init $theta_init \
                                             --m $m \
                                             --sigma $sigma \
                                             --M $M
  done
done
