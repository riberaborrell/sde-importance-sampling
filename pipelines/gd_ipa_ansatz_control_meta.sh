#!/bin/bash

alpha_and_betas=$1
lrs=$2

for i in $alpha_and_betas
do
  IFS=',' read alpha beta <<< "${i}"
  for lr in $lrs
  do
    echo $alpha $beta $lr
    python mds/script_gd_ipa_ansatz_control.py --alpha $alpha \
                                               --beta $beta \
                                               --theta-init meta \
                                               --m 30 \
                                               --epochs-lim 50 \
                                               --lr $lr \
                                               --M 1000
  done
done
