all_libs=(
    "A,C,G,U,GG,CC,GC,GGGG,GCC,GU,CU,GA,GGGGGCC,CCC,GCGC,UC,GGC,GGGGGCCCC,GGCC,CGC,AU,AC,UU,CA,GGGGGCCCCGC,GCU,GGU,GCCCC"
    "A,C,G,U,CU,CGC,GC,UA,GCGGGGGCCCCGCA,GCA,GU,GCGG,UCC,CGCC,CGA,CGCGCGGGGGCCCC,ACC,CCCC,GCGA,UU,GGCU,CGGGG,CA,GCU,GCG,CC,CCGCGCGGGGGCC,AGU,AA"
    "A,C,G,U,GG,CC,GGGG,CCCC,GGGGG,CGCG,GCCCC,GGGGGCCCCGCGCG,GGGGCCCCGCGC,GCG,GCCCCGCG,CG,GGGGCCCC,GGGGGCCCCCC,GGGGGCCCC,GGGGGGGGGCCCCG,GGCCCC,GGGGGGGCCCCGGG,GGGGGGCCCCGC,GGGGGCCCCGCGGG,CCCCGC,CCCCGCGGG,GGCCCCGC,GC"
    "A,C,G,U,CCCC,CCGCGCGG,GC,GCGG,GCGCGG,CCCCCCGCGCGG,CCCCCCGCGCGGGG,CCCGCGCGG,CGCGCGG,GCCGCGCGG,GGGGG,CCCGCGCGGGGGGG,CCCCGCCGCGCGG"
    "A,C,G,U,GG,GGGG,GC,GCGGGG,GGC,GCC,GCGCGGGG,GCGGGGGCC,GCGCGGGGGCC,CC,GCGCGGGGGCCC,CCC,GCGCGGGGGCCCC,GCGGGGGCCCC,GCGCGGGGGCCCCG,CCGCGCGGGGGCCC,GCGCGGGGGCCCCA,GCGGGGGCCCCGC,CGCGCGGGGGCCCC,CCCC,GCGGGGGCCCCGCG,GGCCCC,CCCCC,GCCCC"
    "A,C,G,U,GCGCGGGGGCCCCG,GGGGGG,GCGGGGGCCCC,GA,CGCGGGGGCCCCUU,GCC,GCGCGGGGGCCCCU,GGGGGCC,GGCGCGGGGGCCCC,GGGGGCCCC,GCGCGGGGGCCCCA,GUGU,CGCGGGGGCCCCAA,CGC,GCCG,AAA,CGCGGGGGCCCCGC,GCGGGGGCCCCGU,CGGGGGCCCC,CGCGGGGGCCCCGG,GU,GCGC,GGCC"
)

tasks=(
    L14_RNA1
    L14_RNA2
    L14_RNA3
)
# Function to format list for Hydra CLI
format_for_hydra() {
    local items=$1
    echo "environment.actions=[${items}]"
}


for seed in 1998 2024 42
do
    for lib in "${all_libs[@]}"
    do
        for task in "${tasks[@]}"
        do
            hydra_arg=$(format_for_hydra "$lib")
            sbatch sbatch_scripts/rna_binding/rna_binding.sh \
            experiment=rna_a2c \
            task_name=rna_binding \
            environment.task=${task} \
            environment.output_padding_mask=False \
            environment.modes_path="${HOME}/Chunk-GFN/${task}_modes.pkl" \
            environment.dataset_path="${HOME}/Chunk-GFN/${task}_dataset.pkl" \
            seed=${seed} \
            logger.wandb.name=a2c_${task}_downstream \
            logger.wandb.group=rna_binding \
            $hydra_arg
        done
    done
done
