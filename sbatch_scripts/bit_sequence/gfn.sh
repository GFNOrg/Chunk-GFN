maxlen_cutoff=(
    "120,24,32"
)

experiments=(
    "bit_sequence_prioritized"
#    "bit_sequence_prioritized_chunk"
#    "bit_sequence_prioritized_chunk_replacement"
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
            seed=${seed} \
            environment.max_len=${length} \
            environment.batch_size=${batch_size} \
            environment.threshold=28 \
            algo.reward_temperature=0.2 \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            algo.replay_refactor=backward \
            environment.output_padding_mask=False \
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=bit_sequence
        done
    done
done
