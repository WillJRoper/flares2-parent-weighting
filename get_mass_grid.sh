#!/bin/bash -l
#SBATCH --ntasks 128
#SBATCH -N 1
#SBATCH -J Grid-WITH-PERMISSION
#SBATCH -o logs/standard_output_file.%J.out
#SBATCH -e logs/standard_error_file.%J.err
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

# Run the program 200 times (on 128 cores).
mpirun -np 128 /cosma/home/dp004/dc-rope1/parallel_tasks/build/parallel_tasks 0 200 "python ./grid_parent_distributed.py %d 200"

source deactivate

echo "Job done, info follows..."
sstat --jobs=${SLURM_JOBID}.batch --format=JobID,MaxRSS,AveCPU,AvePages,AveRSS,AveVMSize
exit



