import os
import sys
sys.path.insert(0, '.')
from subprocess import call

JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
offset = 0

job_dict = {
    1+offset: 'TST_data_exp_wo_covariate_IMAVAE.py', 
    2+offset: 'TST_data_exp_wo_covariate_IMAVAE_MLP.py', 
    3+offset: 'TST_data_exp_with_covariate_IMAVAE.py', 
    4+offset: 'TST_data_exp_with_covariate_IMAVAE_MLP.py', 
}

if JOB_ID not in job_dict:
    print(f"{JOB_ID} not here!")
    quit()

call(["python", job_dict[JOB_ID]])