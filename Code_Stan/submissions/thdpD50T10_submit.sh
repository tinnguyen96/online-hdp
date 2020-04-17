#!/bin/bash  
#SBATCH --ntasks-per-node=4 # core count
#SBATCH -o ../logs/thdpD50T10_submit.sh.log-%j
#SBATCH -a 0-4

module load anaconda/2020a 
cd ..
python -u wikipedia.py --method thdp --seed $SLURM_ARRAY_TASK_ID --maxiter 1000 --topiciter 300 --batchsize 50 --K 10 30 50 70 90