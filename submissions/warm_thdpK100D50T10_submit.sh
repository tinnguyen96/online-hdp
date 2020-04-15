#!/bin/bash  
#SBATCH --ntasks-per-node=4 # core count
#SBATCH -o ../logs/warm_thdpK100D50T10_submit.sh.log-%j
#SBATCH -a 2-4

module load anaconda/2020a 
cd ..
python -u wikipedia.py --method thdp --seed $SLURM_ARRAY_TASK_ID --maxiter 3000 --topiciter 100 --batchsize 50 --K 100 --T 10 --topicinfo LDA results/ldaK100_D50_wiki10k_wiki1k/ 100 