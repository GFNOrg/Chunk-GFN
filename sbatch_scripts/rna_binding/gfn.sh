tasks_cutoff=(
    "L14_RNA1,3,11,0.1",
    #"L50_RNA1,10,22,0.0133333",
    #"L100_RNA1,20,5,0.005,0.85,50"
)

algorithms=(
    "rna_prioritized"
    "rna_prioritized_chunk"
    "rna_prioritized_chunk_replacement"
)

modes_path="${HOME}/code/chunkgfn/L14_RNA1_modes.pickle"

#for seed in 1998 2024 42
for seed in 1987 1963 2000
do
    for algo in "${algorithms[@]}"
    do
        for task in "${tasks_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            task="${fields[0]}"
            cutoff="${fields[1]}"
            cutoff=$((cutoff))
            partition_init="${fields[2]}"
            partition_init=$((partition_init))
            temperature="${fields[3]}"
            temperatue=$(echo "$temperature" | bc)

            if [[ "$task" == "L14_RNA1" ]]; then
                dataset_path="${HOME}/code/chunkgfn/L14_RNA1_dataset.pkl"
            else
                dataset_path=null
            fi
            sbatch sbatch_scripts/rna_binding/rna_binding.sh \
            experiment=${algo} \
            task_name=rna_binding \
            seed=${seed} \
            environment.task=${task} \
            environment.output_padding_mask=False \
            environment.modes_path=${modes_path} \
            environment.dataset_path=${dataset_path} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            algo.reward_temperature=${temperature} \
            algo.partition_init=${partition_init} \
            logger.wandb.name=${algo}_${task}_bpe \
            logger.wandb.group=rna_binding

        done
    done
done
