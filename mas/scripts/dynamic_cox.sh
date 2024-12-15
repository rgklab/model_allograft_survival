#!/bin/bash 

#SBATCH --job-name="cox_missing"
#SBATCH -c 10
#SBATCH -w rosetta
#SBATCH --mem=20G

# 10 cpus and 20G for mas models

python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --imputation="mean"
python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --imputation="median"

# python -u -m mas.engine.dynamic_cox.dcox_graft_delta train --outcome="graft" --feature="mas" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="cancer" --feature="full" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="mortality" --feature="mas" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main final_model --outcome="mortality" --feature="mas" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="albi" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="albi" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main final_model --outcome="graft" --feature="mas" --k=0 --acc=0



# below for time delta exps

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=1 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=2 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=2 --acc=1

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=3 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="mas" --k=3 --acc=1


# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=0 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=1 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=2 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=2 --acc=1

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=3 --acc=0

# python -u -m mas.engine.dynamic_cox.main train --outcome="graft" --feature="full" --k=3 --acc=1
