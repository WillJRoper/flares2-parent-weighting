#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-100%4
#SBATCH -p cosma8 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --cpus-per-task=1
#SBATCH -J FLARES-2-Grid #Give it something meaningful.
#SBATCH -o logs/output_hlr_job.%A_%a.out
#SBATCH -e logs/error_hlr_job.%A_%a.err
#SBATCH -t 72:00:00

cd /cosma/home/dp004/dc-rope1/FLARES/FLARES-2-codes/flares2-parent-weighting

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
python grid_parent_distributed.py $i 100

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

