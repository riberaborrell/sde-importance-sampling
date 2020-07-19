#!/bin/bash

alpha_and_betas=$1

for i in $alpha_and_betas
do
  IFS=',' read alpha beta <<< "${i}"
  echo $alpha $beta
  python mds/langevin1d_reference_solution.py --do-plots \
                                              --alpha $alpha \
                                              --beta $beta
done
