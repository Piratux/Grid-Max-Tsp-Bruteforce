#!/bin/bash
#SBATCH -p main
#SBATCH -n8
module load openmpi
mpiCC -std=c++17 -O2 -o bruteforce-matlab-cluster bruteforce-matlab-cluster.cpp
