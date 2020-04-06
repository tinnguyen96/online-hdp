#!/bin/bash  
#SBATCH --ntasks-per-node=4 # core count
#SBATCH -o ../logs/ldaD20_submit.sh.log-%j
#SBATCH -a 0-4

module load anaconda/2020a 
cd ..
python -u wikipedia.py --method lda --inroot wiki10k --heldoutroot wiki1k --seed $SLURM_ARRAY_TASK_ID --maxiter 1000 --batchsize 20 --numtopics 10 20 30 40 50 60 70 80 90 100