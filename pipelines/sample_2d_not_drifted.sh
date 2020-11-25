#!/bin/bash

while getopts a:b:M: flag
do
    case "${flag}" in
        a) alpha=${OPTARG};;
        b) beta=${OPTARG};;
        M) trajectories=${OPTARG};;
    esac
done

for M in $trajectories
do
  echo $alpha $beta $M
  python mds/sample_2d_not_drifted.py --alpha $alpha \
                                      --beta $beta \
                                      --M $M
done
