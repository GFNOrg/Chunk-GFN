tasks=(
    "L14_RNA1,3",
    "L50_RNA1,10",
    "L100_RNA1,20"
)

for seed in 1998 2024 42
do
    for task in "${tasks[@]}" 
    do
        IFS=',' read -r -d '' -a fields <<< "$task"

        task_name="${fields[0]}"
        cutoff="${fields[1]}"
        cutoff=$((cutoff))
        sbatch sbatch_scripts/rna_binding/template.sh seed=${seed} experiment=rna_binding_chunking_prioritized data.task=${task_name} gfn.replay_buffer.cutoff_distance=${cutoff} logger.wandb.name=chunk_prioritized_${task_name}
    done
done