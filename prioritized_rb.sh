maxlen_cutoff=(
    "64,12",
    "128,25",   
)

algorithms=(
    "bit_sequence_chunking_prioritized"
)

for seed in 1
do
    for algo in "${algorithms[@]}"
    do
        for task in "${maxlen_cutoff[@]}"
        do
            for use_pb_kolya in False
            do
                for use_pb_onelookahead in True
                do 
                    IFS=',' read -r -d '' -a fields <<< "$task"

                    length="${fields[0]}"
                    length=$((length))
                    cutoff="${fields[1]}"
                    cutoff=$((cutoff))

                    if [[ $algo == *"chunking"* ]]; then
                        sbatch bit_sequence.sh \
                        experiment=${algo} \
                        task_name=bit_sequence \
                        seed=${seed} \
                        data.max_len=${length} \
                        gfn.chunk_algorithm=bpe \
                        gfn.library_update_frequency=10 \
                        gfn.n_samples=10000 \
                        gfn.replay_buffer.capacity=10000 \
                        gfn.use_pb_kolya=${use_pb_kolya} \
                        gfn.replay_buffer.cutoff_distance=${cutoff} \
                        gfn.reward_temperature=0.3333333333333333 \
                        logger.wandb.name=${algo}_${length}_pblookahead_${use_pb_onelookahead}_bpe \
                        logger.wandb.group=bit_sequence    
                    else
                        sbatch bit_sequence.sh \
                        experiment=${algo} \
                        task_name=bit_sequence \
                        seed=${seed} \
                        data.max_len=${length} \
                        gfn.replay_buffer.cutoff_distance=${cutoff} \
                        gfn.reward_temperature=0.3333333333333333 \
                        logger.wandb.name=${algo}_${length}_bpe \
                        logger.wandb.group=bit_sequence  
                    fi
                done       
            done
        done
    done
done