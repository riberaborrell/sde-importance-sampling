#!/bin/bash

while getopts n:a:b:t:N: flag
do
    case "${flag}" in
        n) n=${OPTARG};;
        a) alpha_i=${OPTARG};;
        b) betas=${OPTARG};;
        t) dt=${OPTARG};;
        N) N=${OPTARG};;
    esac
done

for beta in $betas
do
  echo "n=$n, alpha_i=$alpha_i, beta=$beta, dt=$dt, N=$N"
  python mds/sample_nd_not_controlled.py --n $n \
	  				                     --alpha-i $alpha_i \
                                      	 --beta $beta \
                                      	 --dt $dt \
                                         --N $N
done
