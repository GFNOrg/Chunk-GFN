#!/bin/bash
#SBATCH --job-name=gfn
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
source $VENV/ai_scientist/bin/activate

python main.py trainer.max_epochs=5000