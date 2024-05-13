#!/bin/bash
#SBATCH --job-name=chunkgfn_ccds
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=4-06:00:00
#SBATCH --output /network/scratch/v/vivianoj/chunkgfn/logs/ccds/slurm-%j.out
#SBATCH --error /network/scratch/v/vivianoj/chunkgfn/logs/ccds/slurm-%j.err

eval "$(conda shell.bash hook)"
conda activate chunkgfn

export WANDB_MODE="offline"
exec python main.py "$@"
