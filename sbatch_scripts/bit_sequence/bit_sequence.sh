#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=24G
#SBATCH --time=4-06:00:00
#SBATCH -o /home/mila/l/lena-nehale.ezzine/scratch/chunkgfn/slurm-%j.out

module --quiet load python/3.10
source /home/mila/l/lena-nehale.ezzine/venvs/chunk/bin/activate


exec python main.py "$@"
