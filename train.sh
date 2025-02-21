#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --output=slurm/momaland-train-%j.out
#SBATCH --error=slurm/momaland-train-%j.err
#SBATCH --job-name=rl-train-%j
#SBATCH --gres=gpu:full:1

# Enter your environment path here
PYTHON_ENV="../moma_env"
export WANDB_API_KEY=
source $PYTHON_ENV/bin/activate
python3.9 .\momaland\learning\cooperative_momappo\continuous_momappo.py --envi-id moegridLVFeeder_v0 --exp-name first_test --seed 42 --ref-point 1 1 1 1 1 0 --track --timesteps-per-weight 1000