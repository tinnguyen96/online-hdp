#!/bin/bash  
#SBATCH --ntasks-per-node=6 # core count
#SBATCH -o ../logs/nhdpD50T20_submit.sh.log-%j
#SBATCH -a 0

module load anaconda/2020a 
cd ..
python -u wikipedia.py --method nhdp --seed $SLURM_ARRAY_TASK_ID --batchsize 50 --K 100 --T 20 --LLiter 100 --topiciter 250