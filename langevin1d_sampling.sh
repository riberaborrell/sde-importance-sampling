#!/bin/bash

alphas='1 2'
betas='1 2'

for alpha in $alphas
do
  for beta in $betas
  do
    echo $alpha $beta
    python mds/langevin1d_sampling_not_drifted.py --do-plots \
                                                  --alpha $alpha \
                                                  --beta $beta \
                                                  --M 1000
    python mds/langevin1d_sampling_drifted.py --do-plots \
                                              --alpha $alpha \
                                              --beta $beta \
                                              --M 1000
  done
done
