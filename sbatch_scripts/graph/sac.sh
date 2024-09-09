maxlen_cutoff=(
    "7,3,12"
    "10,5,25"
)

experiments=(
    "graph_sac"
    "graph_sac_chunk"
    "graph_sac_chunk_replacement"
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
            seed=${seed} \
            environment.max_nodes=${length} \
            environment.threshold=${threshold} \
            algo.replay_buffer.cutoff_distance=${cutoff} \
            algo.entropy_coefficient=0.1 \
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=graph
           
        done
    done
done
