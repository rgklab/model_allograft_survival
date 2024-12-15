#!/bin/bash 

#SBATCH --job-name="mean_tdci"
#SBATCH -c 40
#SBATCH -w rosetta
#SBATCH --mem=200G
#SBATCH --gres=gpu:1


# python -u -m mas.engine.main mean_tdci --model="deephit" --outcome="graft" --feature="full" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="deephit" --outcome="graft" --feature="mas" --dataset="test"

# python -u -m mas.engine.main mean_tdci --model="cox" --outcome="graft" --feature="full" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="deepsurv" --outcome="graft" --feature="full" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="forest" --outcome="graft" --feature="full" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="cox" --outcome="graft" --feature="mas" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="deepsurv" --outcome="graft" --feature="mas" --dataset="test"
# python -u -m mas.engine.main mean_tdci --model="forest" --outcome="graft" --feature="mas" --dataset="test"

for i in {1..11}
do
    python -u -m mas.engine.main optn_mean_tdci --model="forest" --outcome="graft" --feature="mas" --dataset="test" --OPTN=$i
done

for i in {1..11}
do
    python -u -m mas.engine.main optn_mean_tdci --model="deephit" --outcome="graft" --feature="mas" --dataset="test" --OPTN=$i
done