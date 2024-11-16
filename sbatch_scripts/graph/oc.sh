maxlen_cutoff=(
    "7,12"
)

experiments=(
    "graph_oc"
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
            threshold="${fields[2]}"
            threshold=$((threshold))

            sbatch sbatch_scripts/graph/graph.sh \
            experiment=${exp} \
            task_name=graph \
            seed=${seed} \
            environment.max_nodes=${length} \
            environment.threshold=${threshold} \
            algo.entropy_coeff=0.05 \
            algo.num_options=10 \
            logger.wandb.name=${exp}_${length}_bpe \
            logger.wandb.group=graph   
        done
    done
done
