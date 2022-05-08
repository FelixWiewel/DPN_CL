#!/bin/bash -l

# Slurm parameters
#SBATCH --chdir=../../../
#SBATCH --job-name=RN32_CIFAR100_UB
#SBATCH --output=log_RN32_CIFAR100_UB_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --qos=ebatch
#SBATCH --partition=empl

# Activate everything you need
module load cuda/11.2
pyenv activate venv
# Run your python code
python -u train_ResNet_rehearsal.py --config="./experiments/CIFAR100_5/RN32_UB"
