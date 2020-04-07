#!/bin/bash  
#SBATCH --ntasks-per-node=6 # core count
#SBATCH -o ../logs/thdpD20T20_submit.sh.log-%j
#SBATCH -a 0

module load anaconda/2020a 
cd ..
python -u wikipedia.py --method thdp --seed $SLURM_ARRAY_TASK_ID --batchsize 20 --K 100 --T 20