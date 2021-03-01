#!/bin/bash

while getopts n:a:b:N: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        a) alpha_is=${OPTARG};;
        b) beta=${OPTARG};;
        N) N=${OPTARG};;
    esac
done

for alpha_i in $alpha_is
do
  echo "n=$n, alpha_i=$alpha_i, beta=$beta, N=$N"
  python mds/sample_nd_not_controlled.py --n $n \
	  				 --alpha-i $alpha_i \
                                      	 --beta $beta \
                                         --N $N
done
