#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
source $VENV/ai_scientist/bin/activate

# Define the tasks
cat <<EOF > $SLURM_TMPDIR/tasks.conf
0 sbatch_scripts/wrapper.sh python main.py seed=1998 logger.wandb.id=${SLURM_JOB_ID}-0 $@
1 sbatch_scripts/wrapper.sh python main.py seed=2024 logger.wandb.id=${SLURM_JOB_ID}-1 $@
2 sbatch_scripts/wrapper.sh python main.py seed=42 logger.wandb.id=${SLURM_JOB_ID}-2 $@
EOF

srun -l --output=$SCRATCH/slurm-%j-%t.out --multi-prog $SLURM_TMPDIR/tasks.conf
