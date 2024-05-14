maxlen_cutoff=(
    "32,6",
    "64,12",
    "128,25"
)

algorithms=(
    "ccds_chunk_prioritized"
    "ccds_chunk_replacement_prioritized"
    "ccds_prioritized"
)

for seed in 1998 2024 42
do
    for algo in "${algorithms[@]}"
    do
        for task in "${maxlen_cutoff[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            length="${fields[0]}"
            length=$((length))
            cutoff="${fields[1]}"
            cutoff=$((cutoff))

            if [[ $algo == *"chunk"* ]]; then
                sbatch sbatch_scripts/ccds/ccds.sh \
                experiment=${algo} \
                task_name=ccds \
                seed=${seed} \
                data.max_len=${length} \
                gfn.chunk_algorithm=bpe \
                gfn.n_samples=10000 \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.3333333333333333 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=ccds
            else
                sbatch sbatch_scripts/ccds/ccds.sh \
                experiment=${algo} \
                task_name=ccds \
                seed=${seed} \
                data.max_len=${length} \
                gfn.replay_buffer.cutoff_distance=${cutoff} \
                gfn.reward_temperature=0.3333333333333333 \
                logger.wandb.name=${algo}_${length}_bpe \
                logger.wandb.group=ccds
            fi
        done
    done
done
