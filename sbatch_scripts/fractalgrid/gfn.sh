hps=(
    "65,195"
    "129,387"
    "257,771"
)

experiments=(
    "fractalgrid_tbgfn"
    "fractalgrid_tbgfn_chunk"
    "fractalgrid_tbgfn_chunk_replacement"
)

#for seed in 1998 2024 42
for seed in 1987 1963 2000
do
    for exp in "${experiments[@]}"
    do
        for hp in "${hps[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$hp"

            side_length="${fields[0]}"
            side_length=$((side_length))
            in_dim="${fields[1]}"
            in_dim=$((in_dim))

            sbatch sbatch_scripts/fractalgrid/fractalgrid.sh \
            experiment=${exp} \
            task_name=fractalgrid \
            seed=${seed} \
            environment.side_length=${side_length} \
            algo.replay_buffer.cutoff_distance=1 \
            algo.forward_policy.in_dim=${in_dim} \
            logger.wandb.name=${exp}_${side_length} \
            logger.wandb.group=fractalgrid

        done
    done
done
