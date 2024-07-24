maxlen_cutoff=(
    "64,12,32",
    "128,25,32"
)

experiments=(
    "bit_sequence_a2c"
    "bit_sequence_a2c_chunk"
    "bit_sequence_a2c_chunk_replacement"
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
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=bit_sequence        
        done
    done
done
