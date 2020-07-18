#!/bin/bash

alphas='1 2'
betas='1 2 4 6'

for alpha in $alphas
do
  for beta in $betas
  do
    echo $alpha $beta
    python mds/langevin1d_reference_solution.py --do-plots \
                                                --alpha $alpha \
                                                --beta $beta
  done
done
