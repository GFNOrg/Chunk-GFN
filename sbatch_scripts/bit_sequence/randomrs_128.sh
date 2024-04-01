#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
module load libffi
module load OpenSSL/1.1
source $VENV/ai_scientist/bin/activate

python main.py seed=42 experiment=bit_sequence_random_rs data.max_len=128 logger.wandb.name="randomrs-len-128"