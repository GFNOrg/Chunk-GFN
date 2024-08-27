#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -o /scratch/jaggbow/fractalgrid/slurm-%j.out


module --quiet load python/3.10
source $HOME/venvs/ai_scientist/bin/activate


exec python main.py logger=wandb_offline "$@"
