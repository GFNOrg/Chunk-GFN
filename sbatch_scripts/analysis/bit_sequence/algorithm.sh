maxlen_cutoff=(
    "32,6",
    "64,12",
    "128,25"
)

algorithms=(
    "bit_sequence_chunk_prioritized"
)

chunking_methods=(
    "bpe",
    "wordpiece",
    "uniform"
)

for seed in 1998 2024 42
do
    for chunk_method in "${chunking_methods[@]}"
    do
        for task in "${maxlen_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            length="${fields[0]}"
            length=$((length))
            cutoff="${fields[1]}"
            cutoff=$((cutoff))

            sbatch sbatch_scripts/analysis/bit_sequence/bit_sequence.sh \
            experiment=bit_sequence_chunk_prioritized \
            task_name=bit_sequence \
            seed=${seed} \
            data.max_len=${length} \
            gfn.chunk_algorithm=${chunk_method} \
            gfn.library_update_frequency=10 \
            gfn.n_samples=10000 \
            gfn.replay_buffer.cutoff_distance=${cutoff} \
            gfn.reward_temperature=0.3333333333333333 \
            logger.wandb.name=bitseq_${chunk_method}_${length} \
            gfn.chunk_algorithm=${chunk_method} \
            logger.wandb.group=bs_chunking_algorithm
            break
        done
    done
    break
done
