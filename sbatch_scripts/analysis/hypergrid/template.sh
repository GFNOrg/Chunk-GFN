#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=50:00:00
#SBATCH -o /scratch/jaggbow/slurm-%j.out
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

module --quiet load python/3.10
source /home/jaggbow/venvs/ai_scientist/bin/activate

export WANDB_RESUME=allow
export WANDB_RUN_ID=$SLURM_JOB_ID
python main.py "$@"
