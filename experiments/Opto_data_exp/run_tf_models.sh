#!/bin/bash
#SBATCH -p carlsonlab-gpu --account=carlsonlab --gres=gpu:1 --mem=64G
#SBATCH --job-name=opto_data_exp
#SBATCH --output=opto_data_exp_%a.out
#SBATCH --error=opto_data_exp_%a.err
#SBATCH -a 101-102
#SBATCH -c 2
#SBATCH --nice

srun singularity exec --nv --bind /work/zj63 /datacommons/carlsonlab/Containers/tfgpu_v4.simg python run_tf_models.py
