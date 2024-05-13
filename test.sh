#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=6-00:00:00
#SBATCH --output /network/scratch/v/vivianoj/chunkgfn/logs/rna_binding/slurm-%j.out
#SBATCH --error /network/scratch/v/vivianoj/chunkgfn/logs/rna_binding/slurm-%j.err

python main.py \
    experiment=rna_binding_chunk_prioritized \
    task_name=rna_binding \
    data.task=L14_RNA1 \
    seed=1234 \
    data.modes_path=/home/mila/v/vivianoj/code/chunkgfn/L14_RNA1_modes.pkl \
    gfn.chunk_algorithm=bpe \
    gfn.library_update_frequency=25 \
    gfn.n_samples=10000 \
    gfn.replay_buffer.cutoff_distance=3 \
    gfn.reward_temperature=0.3333333333333333 \
    logger.wandb.name=ccds_chunk_prioritized_32_bpe \
    logger.wandb.group=ccds \
    trainer=default
