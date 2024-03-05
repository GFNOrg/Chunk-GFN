#!/bin/bash
#SBATCH --job-name=gfn-bs-128
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
module load libffi
module load OpenSSL/1.1
source $VENV/ai_scientist/bin/activate

python main.py seed=42 data=bit_sequence gfn=tb_gfn trainer.max_epochs=1000 data.max_len=128 gfn.replay_buffer.cutoff_distance=25 gfn.reward_temperature=0.3333 logger.wandb.name="prioritized-len-128"
