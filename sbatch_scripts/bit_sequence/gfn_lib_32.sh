#!/bin/bash
#SBATCH --job-name=gfnlib-bs-32
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=17:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurm-%j.out

module --quiet load python/3.10
module load libffi
module load OpenSSL/1.1
source $VENV/ai_scientist/bin/activate

python main.py seed=42 data=bit_sequence trainer.check_val_every_n_epoch=20 gfn.n_trajectories=30 gfn.library_update_frequency=75 gfn=tb_gfn_variable trainer.max_epochs=1000 data.max_len=32 gfn.replay_buffer.cutoff_distance=6 gfn.reward_temperature=0.3333 logger.wandb.name="lib-len-32"
