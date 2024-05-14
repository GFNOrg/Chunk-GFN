#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=6-00:00:00
#SBATCH --output /network/scratch/v/vivianoj/chunkgfn/logs/rna_binding/slurm-%j.out
#SBATCH --error /network/scratch/v/vivianoj/chunkgfn/logs/rna_binding/slurm-%j.err

eval "$(conda shell.bash hook)"
conda activate chunkgfn

export WANDB_MODE="offline"
exec python main.py "$@"

