#!/bin/bash 

#SBATCH --job-name="masforest"
#SBATCH -c 50
#SBATCH -w rosetta
#SBATCH --mem=100G

# python -u -m mas.engine.masforest.sf_search train --feature="mas"
# python -u -m mas.engine.masforest.sf_search train --feature="full"

python -u -m mas.engine.masforest.train train --model="mas" --min_samples_split=20 --max_depth=5
python -u -m mas.engine.masforest.train train --model="full" --min_samples_split=10 --max_depth=5