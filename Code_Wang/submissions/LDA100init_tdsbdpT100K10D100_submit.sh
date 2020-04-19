#!/bin/bash  
#SBATCH --ntasks-per-node=1 # core count
#SBATCH -o ../logs/LDA100init_tdsbdpT100K10D100.sh.log-%j
#SBATCH -a 0-4

module load anaconda/2020a 
cd ..
python -u run_online_hdp.py --random_seed $SLURM_ARRAY_TASK_ID --topic_info LDA results/ldaK100_D50_wiki10k_wiki1k/ 100 