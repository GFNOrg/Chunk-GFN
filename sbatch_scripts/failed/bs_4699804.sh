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
experiment=bit_sequence_chunk_prioritized \
task_name=bit_sequence \
seed=42 \
ckpt_path="/network/scratch/o/oussama.boussif/chunkgfn/logs/bit_sequence/runs/4699804/checkpoints/last.ckpt" \
data.max_len=64 \
gfn.chunk_algorithm=bpe \
gfn.n_samples=10000 \
gfn.replay_buffer.cutoff_distance=12 \
gfn.reward_temperature=0.3333333333333333 \
logger.wandb.name=bit_sequence_chunk_prioritized_64_bpe \
logger.wandb.group=bit_sequence \
logger.wandb.id=4699804 \
logger.wandb.resume=must
