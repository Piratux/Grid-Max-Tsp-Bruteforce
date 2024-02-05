#!/bin/bash
#SBATCH -p main
#SBATCH -n4
module load openmpi
mpiCC -O2 -o test_mpi test_mpi.cpp
mpirun test_mpi
