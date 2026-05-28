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

# Run your Python script
# & makes command run without blocking the script and then we wait for all results
#we need no srun since all jobs run on one reservation

cd cis-HMDMM-95300ps/
whatcat-md  LCC.pdb -l HMD.sdf -t 100 -eqt 1000 -geom "resname HMD and name O4x, resid 131 and name H" -geom "resname HMD and name C15x, resid 130 and name OG" -rt 100 -ncores 16 --platform HIP > LCC_cis-HMD_md.out &

cd ../cis-HNDMM-MTD1-NAC/
whatcat-md LCC.pdb -l HND.sdf -t 100 -eqt 1000 -geom "resname HND and name O4x, resid 131 and name H" -geom "resname HND and name C16x, resid 130 and name OG" -rt 100 -ncores 16 --platform HIP > LCC_cis-HND_md.out &

cd ../trans-HMDMM-MTD1-NAC/
whatcat-md LCC.pdb -l HMD.sdf -t 100 -eqt 1000 -geom "resname HMD and name O4x, resid 131 and name H" -geom "resname HMD and name C15x, resid 130 and name OG" -rt 100 -ncores 16 --platform HIP > LCC_trans-HMD_md.out &

cd ../trans-HNDMM-MTD1-NAC/
whatcat-md LCC.pdb -l HND.sdf -t 100 -eqt 1000 -geom "resname HND and name O4x, resid 131 and name H" -geom "resname HND and name C16x, resid 130 and name OG" -rt 100 -ncores 16 --platform HIP > LCC_trans-HND_md.out &

wait

#END OF SCRIPT
