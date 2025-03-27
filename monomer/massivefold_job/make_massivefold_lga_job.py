import os
target_dir = '/data/data1/conglab/qcong/CASP16/stage1_monomer_inputs/targets/'
massivefold_dir = '/data/data3/conglab/jzhan6/massivefold/CASP16-CAPRI/'
targets = [file for file in os.listdir(target_dir) 
           if file.startswith('T1') and file.endswith('.pdb') 
           and '_' not in file and 'v2' not in file]

targets.sort()

for target in targets:
    job_str = f"""#!/bin/bash
#SBATCH -p 256GB
#SBATCH -n 48
#SBATCH -t 28-00:00:00
#SBATCH -o /scratch/job_{target}.log
#SBATCH -e job_{target}.err
#SBATCH -J job_{target}
#SBATCH --mem=250GB

ulimit -s unlimited
python /home2/s439906/project/CASP16/monomer/massivefold_lga.py --target {target}
"""
    target_id = target.split('.')[0]
    with open(f'{target_id}_lga_job', 'w') as f:
        f.write(job_str)
    os.system(f'sbatch {target_id}_lga_job')