tasks_cutoff=(
    "L14_RNA1,3",
    "L50_RNA1,10",
    "L100_RNA1,20"
)

experiments=(
    "rna_random_sampler"
    "rna_random_sampler_chunk"
    "rna_random_sampler_chunk_replacement"
)

modes_path="${HOME}/Chunk-GFN/L14_RNA1_modes.pkl"

for seed in 1998 2024 42
do
    for exp in "${experiments[@]}"
    do
        for task in "${tasks_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            task="${fields[0]}"
            cutoff="${fields[1]}"
            cutoff=$((cutoff))

            if [[ "$task" == "L14_RNA1" ]]; then
                dataset_path="${HOME}/Chunk-GFN/L14_RNA1_dataset.pkl"
            else
                dataset_path=null
            fi

            sbatch sbatch_scripts/rna_binding/rna_binding.sh \
            experiment=${exp} \
            task_name=rna_binding \
            environment.task=${task} \
            environment.output_padding_mask=False \
            environment.modes_path=${modes_path} \
            environment.dataset_path=${dataset_path} \
            seed=${seed} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            logger.wandb.name=${exp}_${task} \
            logger.wandb.group=rna_binding

        done
    done
done