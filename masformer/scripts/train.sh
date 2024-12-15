#!/bin/bash
#SBATCH --job-name="masformer"
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --output=/dev/null
#SBATCH --error=slurm-%j.err

# put your command here
# python -m masformer.engine.train final --out_dir='mas_rtx' --feature="mas"

# python -m masformer.engine.rnn.train final --out_dir='full_t4v2' --feature="full"

python -m masformer.engine.rnn.train final --out_dir='mas_rtx' --feature="mas"