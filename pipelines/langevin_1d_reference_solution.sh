#!/bin/bash

while getopts a:b:h: flag
do
    case "${flag}" in
        a) alphas=${OPTARG};;
        b) betas=${OPTARG};;
        h) h=${OPTARG};;
    esac
done

for alpha in $alphas
do
  for beta in $betas
  do
  echo $alpha $beta $h
  python mds/langevin_1d_reference_solution.py --alpha $alpha \
                                               --beta $beta \
                                               --h $h
  done
done
