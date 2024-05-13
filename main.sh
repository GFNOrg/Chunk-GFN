#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=4-06:00:00
#SBATCH -o /home/mila/l/lena-nehale.ezzine/scratch/chunkgfn/slurm-%j.out

module --quiet load python/3.10
source ~/venvs/chunk/bin/activate


exec python main.py experiment=bit_sequence_chunking_prioritized.yaml task_name=bit_sequence seed=1 data.max_len=64 gfn.chunk_algorithm=bpe gfn.n_samples=10000 gfn.use_pb_kolya=True gfn.replay_buffer.cutoff_distance=12 gfn.reward_temperature=0.3333333333333333 logger.wandb.name=bit_sequence_chunking_prioritized_64_True_bpe logger.wandb.group=bit_sequence