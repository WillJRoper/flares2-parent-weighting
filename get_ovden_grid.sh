#!/bin/bash -l
#SBATCH --ntasks 256
#SBATCH -N 2
#SBATCH --array=1-20%1
#SBATCH -J FLARES2-OVDEN-GRID-L2800N5040
#SBATCH -o logs/L2800N5040.%J.out
#SBATCH -e logs/L2800N5040.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 3:00:00

module purge
#load the modules used to build your program.
module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load pythonconda3/4.5.4

cd /cosma/home/dp004/dc-rope1/FLARES/FLARES-2-codes/flares2-parent-weighting

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

mpirun -np 256 python grid_parent_distributed.py $i

source deactivate

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit



