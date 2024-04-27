#!/bin/bash
#SBATCH -p carlsonlab-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=opto_data_exp
#SBATCH --output=opto_data_exp_%a.out
#SBATCH --error=opto_data_exp_%a.err
#SBATCH -a 1-4
#SBATCH -c 2
#SBATCH --nice

srun singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/multimodal_gp.simg python run_torch_models.py