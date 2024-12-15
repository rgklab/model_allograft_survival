#!/bin/bash 

#SBATCH --job-name="cox_mice"
#SBATCH -c 10
#SBATCH -w rosetta
#SBATCH --mem=20G


# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --imputation="mean"
# python -u mas/data/load_data_dynamic.py

for i in {0..4}
do
    python -u -m mas.engine.dynamic_cox.main train_mice --outcome="graft" --feature="mas" --imputation="mice_gbdt" --dataset=$i
done