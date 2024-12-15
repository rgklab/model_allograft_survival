#!/bin/bash 

#SBATCH --job-name="mean_tdci"
#SBATCH -c 20
#SBATCH -w rosetta
#SBATCH --mem=100G
#SBATCH --gres=gpu:0


# python -u -m mas.engine.main get_x_year_gf --model="cox" --outcome="graft" --feature="mas" --dataset="test"
# python -u -m mas.engine.main get_x_year_gf --model="cox" --outcome="graft" --feature="meld" --dataset="test"
# python -u -m mas.engine.main get_x_year_gf --model="cox" --outcome="graft" --feature="meaf" --dataset="test"
# python -u -m mas.engine.main get_x_year_gf --model="cox" --outcome="graft" --feature="albi" --dataset="test"
python -u -m mas.engine.main get_x_year_gf --model="meld" --outcome="graft" --feature="meld" --dataset="test"
python -u -m mas.engine.main get_x_year_gf --model="albi" --outcome="graft" --feature="albi" --dataset="test"
