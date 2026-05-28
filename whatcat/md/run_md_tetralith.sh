#!/bin/bash
#SBATCH -p tetralith
#SBATCH -A naiss2025-5-398

#unblock for dev jobs shorter than 1h
##SBATCH --reservation=now

#resource allocation
#do not change mem-per-cpu without requesting a fat node
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=2500
#SBATCH -t 1:00:00

#unblock for exclusive access to the node, necessary for large disk/RAM
##SBATCH --exclusive
#unblock for fat node
##SBATCH -C fat
#unblock for fat large disk node (for Coupled cluster). diskL has a GPU :(
##SBATCH -C 'fat&diskM'

#Tempname
#SBATCH -J whatcat_job

INPUT_NAME="${1%.*}+${3%.*}"

# Rename the job dynamically after submission
scontrol update JobName="${INPUT_NAME}_md" jobid=$SLURM_JOB_ID

module load Miniforge/24.7.1-2-hpc1
conda activate whatcat

# Run your Python script
"whatcat-md" "$@" > "${INPUT_NAME}_md.out"

#END OF SCRIPT
