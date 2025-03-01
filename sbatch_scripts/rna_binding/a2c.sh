tasks_cutoff=(
    "L14_RNA1,3,0.9,0.05",
    #"L50_RNA1,10,0.9,0.005",
    #"L100_RNA1,20,0.85,0.0025"
)

algorithms=(
    "rna_a2c"
    "rna_a2c_chunk"
    "rna_a2c_chunk_replacement"
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
            threshold="${fields[2]}"
            threshold=$(echo "$threshold" | bc)
            entropy_coeff="${fields[3]}"
            entropy_coeff=$(echo "$entropy_coeff" | bc)

            if [[ "$task" == "L14_RNA1" ]]; then
                dataset_path="${HOME}/code/chunkgfn/L14_RNA1_dataset.pickle"
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
            environment.high_reward_threshold=${threshold} \
            algo.entropy_coeff=${entropy_coeff} \
            logger.wandb.name=${algo}_${task} \
            logger.wandb.group=rna_binding

        done
    done
done
