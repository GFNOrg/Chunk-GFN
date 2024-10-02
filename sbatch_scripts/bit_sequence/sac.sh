maxlen_cutoff=(
    "64,8,64,8,0.01"
    "128,16,64,16,0.005"
)

experiments=(
    "bit_sequence_sac"
    "bit_sequence_sac_chunk"
    "bit_sequence_sac_chunk_replacement"
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
            entropy_coeff="${fields[4]}"
            entropy_coeff=$(echo "$entropy_coeff" | bc)

            sbatch sbatch_scripts/bit_sequence/bit_sequence.sh \
            experiment=${exp} \
            task_name=bit_sequence \
            seed=${seed} \
            environment.max_len=${length} \
            environment.threshold=${threshold} \
            environment.batch_size=${batch_size} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            algo.entropy_coefficient=${entropy_coeff} \
            environment.output_padding_mask=False \
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=bit_sequence \
            logger=wandb_offline
           
        done
    done
done
