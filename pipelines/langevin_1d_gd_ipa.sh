#!/bin/bash

while getopts a:b:t:m:s:l:e:M: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        m) m=${OPTARG};;
        s) sigmas=${OPTARG};;
        t) theta_init=${OPTARG};;
        l) lrs=${OPTARG};;
        e) epochs=${OPTARG};;
        M) M=${OPTARG};;
    esac
done

for sigma in $sigmas
do
  for lr in $lrs
  do
    echo $alpha $beta $m $sigma $theta_init $M $lr $epochs
    python mds/gd_1d_ipa_gaussian_ansatz.py --alpha $alpha \
                                            --beta $beta \
                                            --m $m \
                                            --sigma $sigma \
                                            --theta-init $theta_init \
                                            --M $M \
                                            --lr $lr \
                                            --epochs-lim $epochs
  done
done
