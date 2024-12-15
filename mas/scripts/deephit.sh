#!/bin/bash 

#SBATCH --job-name="dhit_full"
#SBATCH -c 30
#SBATCH -w rosetta
#SBATCH --mem=30G
#SBATCH --gres=gpu:1


# python -u -m mas.engine.deephit.main run_agent --sweep_id="" --project_name="deephit" --count=200 --feature="full"
# python -u -m mas.engine.deephit.main run_agent --sweep_id="" --project_name="deephit" --count=200 --feature="mas"

python -u -m mas.engine.deephit.main final_model --feature="full"