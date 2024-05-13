#!/bin/bash

python ../../main.py \
    experiment=ccds_chunk_prioritized \
    task_name=ccds \
    seed=1234 \
    data.max_len=32 \
    gfn.chunk_algorithm=bpe \
    gfn.n_samples=10000 \
    gfn.replay_buffer.cutoff_distance=6 \
    gfn.reward_temperature=0.3333333333333333 \
    logger.wandb.name=ccds_chunk_prioritized_32_bpe \
    logger.wandb.group=ccds \
    trainer=default
