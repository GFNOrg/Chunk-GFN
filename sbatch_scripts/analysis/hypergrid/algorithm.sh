tasks=(
    "16,8",
    "32,8",
    "64,8"
)

for seed in 1998 2024 42 123 5
do
    for algorithm in "uniform" "bpe" "wordpiece"
    do
        for task in "${tasks[@]}"
        do
            IFS=',' read -r -d '' -a fields <<< "$task"

            length="${fields[0]}"
            length=$((length))
            dim="${fields[1]}"
            dim=$((dim))
            in_dim=$((length * (dim+1)))
            n_actions=$((dim+1))
            sbatch sbatch_scripts/analysis/hypergrid/template.sh task_name=algorithm logger.wandb.group=hypergrid_algorithm seed=${seed} gfn.chunk_algorithm=${algorithm} experiment=hypergrid_chunking_prioritized gfn.replay_buffer.cutoff_distance=2 data.ndim=${dim} data.side_length=${length} gfn.action_model.n_primitive_actions=${n_actions} gfn.forward_model.in_dim=${in_dim} logger.wandb.name=chunk_${task_name}_${algorithm}
        done
    done
done
