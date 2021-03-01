#!/bin/bash

while getopts n:a:b:N: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        a) alpha_i=${OPTARG};;
        b) betas=${OPTARG};;
        N) N=${OPTARG};;
    esac
done

for beta in $betas
do
  echo "n=$n, alpha_i=$alpha_i, beta=$beta, N=$N"
  python mds/sample_nd_not_controlled.py --n $n \
	  				 --alpha-i $alpha_i \
                                      	 --beta $beta \
                                         --N $N
done
