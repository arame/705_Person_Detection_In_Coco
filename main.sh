#! /bin/bash

#SBATCH --job-name="Geoff P"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=geoffrey.payne@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.output
#SBATCH --error jo%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

module load python/3.7.9

python main.py
