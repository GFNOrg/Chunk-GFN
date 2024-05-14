#!/bin/bash
#SBATCH --job-name=gfn-lib-bpe
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
module load libffi
module load OpenSSL/1.1
source $VENV/ai_scientist/bin/activate

python main.py seed=42 gfn=cond_tb_gfn_variable trainer.max_epochs=1000 data.max_len=30 gfn.library_update_frequency=25 gfn.reward_temperature=0.1 logger.wandb.name="library_T_0.1_len_30"
