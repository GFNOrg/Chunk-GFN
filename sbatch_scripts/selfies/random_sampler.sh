length_cutoff=(
    "10,1",
    "20,2",
    "40,4"
)

algorithms=(
    "random_sampler"
    "random_sampler_chunk"
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
                gfn=${algo} \
                data=selfies_sequence \
                task_name=selfies \
                seed=${seed} \
                data.max_len=${length} \
                gfn.chunk_algorithm=bpe \
                gfn.library_update_frequency=25 \
                gfn.n_samples=10000 \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.2 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=selfies \
                trainer.max_epochs=1500     
            else
                sbatch sbatch_scripts/selfies/selfies.sh \
                gfn=${algo} \
                data=selfies_sequence \
                task_name=selfies \
                seed=${seed} \
                data.max_len=${length} \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.2 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=selfies \
                trainer.max_epochs=1500   
            fi        
        done
    done
done
