#!/bin/bash

while getopts a:b:t:s:l: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        t) theta_init=${OPTARG};;
        s) sigmas=${OPTARG};;
        l) lrs=${OPTARG};;
    esac
done

for sigma in $sigmas
do
  for lr in $lrs
  do
    echo $sigma $lr
    python mds/script_gd_ipa_ansatz_value_f.py --alpha $alpha \
                                               --beta $beta \
                                               --theta-init $theta_init \
                                               --M 2000 \
                                               --m 30 \
                                               --sigma $sigma \
                                               --lr $lr \
                                               --epochs-lim 50
  done
done
