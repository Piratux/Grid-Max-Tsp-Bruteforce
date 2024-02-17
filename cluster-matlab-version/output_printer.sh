#!/bin/bash

# Expected shell script that launches program via mpirun.
mpi_script=$1

# Launch sbatch job.
sbatch_output=$(sbatch ${mpi_script})

# Print sbatch output
echo "${sbatch_output}"

# Get sbatch job id
job_id=$(echo "${sbatch_output}" | awk '{print $4}')

# Save output file
output_file="slurm-${job_id}.out"

# Wait until file appears
until [ -f ${output_file} ]
do
     sleep 1
done

# Continuously print output in real-time
tail -f -n +1 ${output_file} &
