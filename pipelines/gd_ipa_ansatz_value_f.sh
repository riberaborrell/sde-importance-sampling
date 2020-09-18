#!/bin/bash

while getopts a:b:t:m:s:l:e:M: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        t) theta_init=${OPTARG};;
        m) m=${OPTARG};;
        s) sigmas=${OPTARG};;
        l) lrs=${OPTARG};;
        e) epochs=${OPTARG};;
        M) M=${OPTARG};;
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
                                               --m $m \
                                               --sigma $sigma \
                                               --lr $lr \
                                               --epochs-lim $epochs \
                                               --M $M
  done
done
