#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-500
#SBATCH -p cosma6 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --cpus-per-task=1
#SBATCH -J FLARES-2-Grid #Give it something meaningful.
#SBATCH -o logs/output_hlr_job.%A_%a.out
#SBATCH -e logs/error_hlr_job.%A_%a.err
#SBATCH -t 72:00:00

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
./grid_parent_distributed.py $i 500

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

