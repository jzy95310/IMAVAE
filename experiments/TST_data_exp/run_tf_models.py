import os
import sys
sys.path.insert(0, '.')
from subprocess import call

JOB_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
offset = 100

job_dict = {
    1+offset: 'TST_data_exp_wo_covariate_shallow_LSEM.py', 
    2+offset: 'TST_data_exp_wo_covariate_deep_LSEM.py', 
    3+offset: 'TST_data_exp_wo_covariate_svr_LSEM.py', 
    4+offset: 'TST_data_exp_with_covariate_shallow_LSEM.py', 
    5+offset: 'TST_data_exp_with_covariate_deep_LSEM.py', 
    6+offset: 'TST_data_exp_with_covariate_svr_LSEM.py', 
}

if JOB_ID not in job_dict:
    print(f"{JOB_ID} not here!")
    quit()

call(["python", job_dict[JOB_ID]])