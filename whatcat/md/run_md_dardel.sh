#!/bin/bash
#SBATCH -A naiss2025-5-398

#SBATCH -p gpu

#resource allocation
#SBATCH -N 1
#SBATCH -t 24:00:00

#Tempname
#SBATCH -J whatcat_job

#Load modules
ml cray-python/3.11.7
ml rocm/6.3.3
ml craype-accel-amd-gfx90a
ml PDC miniconda3

source activate whatcat

INPUT_NAME="${1%.*}+${3%.*}"

# Rename the job dynamically after submission
scontrol update JobName="${INPUT_NAME}_md" jobid=$SLURM_JOB_ID

# Run your Python script
"whatcat-md" "$@" --platform HIP > "${INPUT_NAME}_md.out"

#END OF SCRIPT
