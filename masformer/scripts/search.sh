#!/bin/bash
#SBATCH --job-name="sweep"
#SBATCH --partition=p100,t4v1,t4v2,rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --output=/dev/null
#SBATCH --error=slurm-%j.err

# put your command here

# python -m masformer.engine.train run_agent --run_name $i --project_name "masformer" --sweep_id ""
# python -m masformer.engine.train run_agent --run_name $i --project_name "masformer" --sweep_id ""
python -m masformer.engine.rnn.train run_agent --run_name $i --project_name "gru" --sweep_id ""
# python -m masformer.engine.rnn.train run_agent --run_name $i --project_name "gru" --sweep_id ""