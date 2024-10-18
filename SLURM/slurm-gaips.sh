#!/bin/sh
# shellcheck disable=SC2206

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

###SBATCH --mincpus=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=shard:6
#SBATCH --mem=16G

#SBATCH --job-name=alphazero


#CUDA_VISIBLE_DEVICES=-1 python Run.py --training-preset 5 --name "not_a_name"
CUDA_VISIBLE_DEVICES=-1 python Run.py --training-preset 1
#CUDA_VISIBLE_DEVICES=-1 python Run.py --training-preset 2
#CUDA_VISIBLE_DEVICES=-1 python Run.py --testing-preset 0



