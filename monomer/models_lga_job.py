import os
import sys

current_dir = os.getcwd()
home_dir = "./model_lga_job/"
job_dirs = ['best_job/', 'first_job/', 'first_of_best_job/']
for job_dir in job_dirs:
    input_dir = os.path.join(home_dir, job_dir, 'MOL2')
    if len(os.listdir(input_dir)) !=74:
        print(f"{input_dir} does not have 74 files, please check")
        sys.exit(1)

    with open(os.path.join(home_dir, job_dir, "lga_cmd"), 'w') as f:
        shebang=f"""#!/bin/bash
#SBATCH -p 256GB
#SBATCH -n 48
#SBATCH -t 28-00:00:00
#SBATCH -o {job_dir[:-1]}.log
#SBATCH -e {job_dir[:-1]}.err
#SBATCH -J {job_dir[:-1]}
#SBATCH --mem=250GB"""
        f.write(shebang)
        f.write('\n')
        f.write('\n')
        f.write('ulimit -s unlimited\n')
        for file in os.listdir(input_dir):
            f.write(f'/data/data1/conglab/jzhan6/software/LGA_package_src/lga -3  -ie  -o1  -sda  -d:4  -gdc_sc {file}\n')
        os.chdir(os.path.join(home_dir, job_dir))
        os.system(f'sbatch lga_cmd')
        os.chdir(current_dir)

