#!/bin/bash

while getopts a:b:m:s:t:l:e: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        m) m=${OPTARG};;
        s) sigma=${OPTARG};;
        t) theta_init=${OPTARG};;
        l) lrs=${OPTARG};;
        e) epochs=${OPTARG};;
    esac
done

for lr in $lrs
do
  echo $alpha $beta $m $sigma $theta_init $M $lr $epochs
  python mds/gd_1d_ipa_gaussian_ansatz.py --alpha $alpha \
                                          --beta $beta \
                                          --m $m \
                                          --sigma $sigma \
                                          --theta-init $theta_init \
                                          --M 1000 \
                                          --lr $lr \
  					  --epochs-lim $epochs

  python mds/sample_1d_drifted.py --alpha $alpha \
                                  --beta $beta \
                                  --m $m \
                                  --sigma $sigma \
                                  --theta gd \
                                  --theta-init $theta_init \
                                  --lr $lr \
                                  --M 10000
done
