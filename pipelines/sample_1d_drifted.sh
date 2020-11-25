#!/bin/bash

while getopts a:b:m:s:t:l:M: flag
do
    case "${flag}" in
        a) alphas=${OPTARG};;
        b) betas=${OPTARG};;
        m) m=${OPTARG};;
        s) sigma=${OPTARG};;
        t) theta=${OPTARG};;
        M) M=${OPTARG};;
    esac
done

for alpha in $alphas
do
  for beta in $betas
  do
  echo $alpha $beta $m $sigma $theta $M
  python mds/sample_1d_drifted.py --alpha $alpha \
                                  --beta $beta \
                                  --m $m \
                                  --sigma $sigma \
                                  --theta $theta \
                                  --M $M
  done
done
