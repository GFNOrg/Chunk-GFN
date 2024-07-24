maxlen_cutoff=(
    "120,24,32"
)

experiments=(
    "bs_random_sampler"
    "bs_random_sampler_chunk"
    "bs_random_sampler_chunk_replacement"
)

for seed in 1998 2024 42
do
    for exp in "${experiments[@]}"
    do
        for task in "${maxlen_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            length="${fields[0]}"
            length=$((length))
            cutoff="${fields[1]}"
            cutoff=$((cutoff))
            batch_size="${fields[2]}"
            batch_size=$((batch_size))

            sbatch sbatch_scripts/bit_sequence/bit_sequence.sh \
            experiment=${exp} \
            task_name=bit_sequence \
            environment.max_len=${length} \
            environment.threshold=28 \
            environment.batch_size=${batch_size} \
            environment.output_padding_mask=False \
            seed=${seed} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            logger.wandb.name=${exp}_${length} \
            logger.wandb.group=bit_sequence

        done
    done
done