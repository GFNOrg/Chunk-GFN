#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=4-06:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
source $VENV/ai_scientist/bin/activate


python main.py \
experiment=bit_sequence_chunk_replacement_prioritized \
task_name=bit_sequence \
seed=2024 \
data.max_len=32 \
gfn.chunk_algorithm=bpe \
gfn.n_samples=10000 \
gfn.replay_buffer.cutoff_distance=6 \
gfn.reward_temperature=0.3333333333333333 \
logger.wandb.name=bit_sequence_chunk_replacement_prioritized_32_bpe \
logger.wandb.group=bit_sequence
