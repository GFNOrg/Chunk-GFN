#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=1-08:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out
#SBATCH --exclude=cn-a006,cn-a010,cn-c032,cn-c024

module --quiet load python/3.10
source $VENV/ai_scientist/bin/activate


exec python main.py "$@"

