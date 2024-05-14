#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
source $VENV/ai_scientist/bin/activate


exec python main.py "$@"
