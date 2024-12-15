#!/bin/bash 

#SBATCH --job-name="mean_tdci"
#SBATCH -c 50
#SBATCH -w rosetta
#SBATCH --mem=200G
#SBATCH --gres=gpu:1


# python -u -m masformer.engine.rnn.evaluate mean_tdci --feature="mas" --dataset="test"
python -u -m masformer.engine.rnn.evaluate mean_tdci --feature="full" --dataset="test"
python -u -m masformer.engine.evaluate mean_tdci --feature="mas" --dataset="test"
python -u -m masformer.engine.evaluate mean_tdci --feature="full" --dataset="test"

# for i in {1..11}
# do
#     python -u -m masformer.engine.evaluate optn_mean_tdci --feature="mas" --dataset="test" --OPTN=$i
# done