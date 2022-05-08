#!/bin/bash -l

# Slurm parameters
#SBATCH --chdir=../../../
#SBATCH --job-name=RN18_SubImageNet_RD_BOS
#SBATCH --output=log_RN18_SubImageNet_RD_BOS_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --gpus=rtx_a6000:1
#SBATCH --qos=ebatch
#SBATCH --partition=empl

# Activate everything you need
module load cuda/11.2
pyenv activate venv
# Run your python code
python -u train_ResNet_rehearsal.py --config="./experiments/SubImageNet_10/RN18_RD_BOS"
