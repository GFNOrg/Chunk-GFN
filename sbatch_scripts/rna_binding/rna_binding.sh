#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-08:00:00
#SBATCH -o /home/mila/v/vivianoj/scratch/logs/chunkgfn/slurm-%j.out
#SBATCH --exclude=cn-a006,cn-a010,cn-c032,cn-c024

source /home/mila/v/vivianoj/miniconda3/bin/activate
conda activate chunkgfn
cd /home/mila/v/vivianoj/code/chunkgfn


exec python main.py "$@"

