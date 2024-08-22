maxlen_cutoff=(
#    "64,12,64,10,0.01,112"
    "128,25,64,20,0.005,226"
)

experiments=(
#    "bit_sequence_tbgfn"
    "bit_sequence_tbgfn_chunk"
    "bit_sequence_tbgfn_chunk_replacement"
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
            threshold="${fields[3]}"
            threshold=$((threshold))
            temperatue="${fields[4]}"
            temperatue=$(echo "$temperatue" | bc)
            partition="${fields[5]}"
            partition=$((partition))

            sbatch sbatch_scripts/bit_sequence/bit_sequence.sh \
            experiment=${exp} \
            task_name=bit_sequence \
            seed=${seed} \
            environment.max_len=${length} \
            environment.threshold=${threshold} \
            environment.batch_size=${batch_size} \
            algo.reward_temperature=${temperatue} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            algo.partition_init=${partition} \
            algo.replay_refactor=backward \
            environment.output_padding_mask=False \
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=bit_sequence
        done
    done
done
