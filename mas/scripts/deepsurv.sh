#!/bin/bash 


#SBATCH --job-name="dsurv_mas"
#SBATCH -c 1
#SBATCH -w rosetta
#SBATCH --mem=15G
#SBATCH --gres=gpu:1

# python -u -m mas.engine.deepsurv.main train --outcome="mortality" --feature="mas"

python -u -m mas.engine.deepsurv.main final_model --outcome="mortality" --feature="mas"
# python -u -m mas.engine.deepsurv.main train --feature="mas"