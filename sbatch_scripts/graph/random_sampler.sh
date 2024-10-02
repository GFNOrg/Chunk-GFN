maxlen_cutoff=(
    "7,3,12"
    "10,5,25"
)

experiments=(
    "graph_random"
    "graph_random_chunk"
    "graph_random_chunk_replacement"
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
            threshold="${fields[2]}"
            threshold=$((threshold))

            sbatch sbatch_scripts/graph/graph.sh \
            experiment=${exp} \
            task_name=graph \
            environment.max_nodes=${length} \
            environment.threshold=${threshold} \
            seed=${seed} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            logger.wandb.name=${exp}_${length} \
            logger.wandb.group=graph

        done
    done
done