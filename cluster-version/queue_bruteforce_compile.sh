#!/bin/bash
#SBATCH -p main
#SBATCH -n5
module load openmpi
mpiCC -std=c++17 -O2 -o bruteforce bruteforce.cpp
