#!/bin/bash

alpha_and_betas=$1
M=$2

for i in $alpha_and_betas
do
  IFS=',' read alpha beta <<< "${i}"
  echo $alpha $beta
  python mds/langevin1d_sampling_not_drifted.py --do-plots \
                                                --alpha $alpha \
                                                --beta $beta \
                                                --M $M
  python mds/langevin1d_sampling_drifted.py --do-plots \
                                            --alpha $alpha \
                                            --beta $beta \
                                            --M $M
done
