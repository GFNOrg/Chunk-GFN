tasks_cutoff=(
    "L14_RNA1,3",
    "L50_RNA1,10",
    "L100_RNA1,20"
)

algorithms=(
    "rna_binding_chunk_prioritized"
    "rna_binding_chunk_replacement_prioritized"
    "rna_binding_prioritized"
)

for seed in 1998 2024 42
do
    for algo in "${algorithms[@]}"
    do
        for task in "${tasks_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            task="${fields[0]}"
            cutoff="${fields[1]}"
            cutoff=$((cutoff))

            modes_path="/home/mila/o/oussama.boussif/Chunk-GFN/L14_RNA1_modes.pkl"

            if [[ $algo == *"chunk"* ]]; then
                sbatch sbatch_scripts/rna_binding/rna_binding.sh \
                logger=wandb_offline \
                experiment=${algo} \
                task_name=rna_binding \
                seed=${seed} \
                data.task=${task} \
                data.modes_path=${modes_path} \
                gfn.chunk_algorithm=bpe \
                gfn.n_samples=10000 \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.125 \
                logger.wandb.name=${algo}_${task}_bpe \
                logger.wandb.group=rna_binding    
            else
                sbatch sbatch_scripts/rna_binding/rna_binding.sh \
                logger=wandb_offline \
                experiment=${algo} \
                task_name=rna_binding \
                seed=${seed} \
                data.task=${task} \
                data.modes_path=${modes_path} \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.125 \
                logger.wandb.name=${algo}_${task}_bpe \
                logger.wandb.group=rna_binding  
            fi             
        done
    done
done
