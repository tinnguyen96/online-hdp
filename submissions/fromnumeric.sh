#!/bin/bash  
#SBATCH --ntasks-per-node=6 # core count
#SBATCH -o ../logs/fromnumeric.sh.log-%j
#SBATCH -a 0

module load anaconda/2020a 
cd ..
python -u wikipedia.py --tau 1.0 --kappa 0.5 --test False --inroot cleanwiki10k