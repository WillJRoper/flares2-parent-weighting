#!/bin/bash -l
#SBATCH --ntasks 512
#SBATCH --array=1-20%4
#SBATCH --cpus-per-task=1
#SBATCH -J FLARES2-REGION-GRID-L2800N5040
#SBATCH -o logs/L2800N5040_regions.%J.out
#SBATCH -e logs/L2800N5040_regions.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 5:00:00

module purge
#load the modules used to build your program.
module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load pythonconda3/4.5.4

cd /cosma/home/dp004/dc-rope1/FLARES/FLARES-2-codes/flares2-parent-weighting

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

mpirun -np 512 python smoothed_grid.py $i L2800N5040 HYDRO
mpirun -np 512 python smoothed_grid.py $i L2800N5040 DMO

source deactivate

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit



