length_cutoff=(
    "10,1",
    "20,2",
    "40,4"
)

algorithms=(
    "selfies_chunk_prioritized_a2c"
    "selfies_prioritized_a2c"
)

for seed in 1998 2024 42
do
    for algo in "${algorithms[@]}"
    do
        for task in "${length_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            length="${fields[0]}"
            length=$((length))
            cutoff="${fields[1]}"
            cutoff=$((cutoff))

            if [[ $algo == *"chunk"* ]]; then
                sbatch sbatch_scripts/selfies/selfies.sh \
                experiment=${algo} \
                task_name=selfies \
                seed=${seed} \
                data.max_len=${length} \
                gfn.chunk_algorithm=bpe \
                gfn.library_update_frequency=25 \
                gfn.n_samples=10000 \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.2 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=selfies    
            else
                sbatch sbatch_scripts/selfies/selfies.sh \
                experiment=${algo} \
                task_name=selfies \
                seed=${seed} \
                data.max_len=${length} \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.2 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=selfies  
            fi             
        done
    done
done
