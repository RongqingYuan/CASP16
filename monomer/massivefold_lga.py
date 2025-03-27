import os
import sys
import argparse
from multiprocessing import Pool

three_to_one = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
target_dir = '/data/data1/conglab/qcong/CASP16/stage1_monomer_inputs/targets/'
massivefold_dir = '/data/data3/conglab/jzhan6/massivefold/CASP16-CAPRI/'
targets = [file for file in os.listdir(target_dir) 
           if file.startswith('T1') and file.endswith('.pdb') 
           and '_' not in file and 'v2' not in file]
targets.sort()

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, help='target pdb file')
args = parser.parse_args()
target = args.target



def get_target_seq_and_mapping(pdb_file):
    seq = ''
    mapping = []
    resnum_to_res = {}
    target_lines = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                target_lines.append(line)
                resname_three = line[17:20].strip()
                resname_one = three_to_one[resname_three]
                chain_id = line[21]
                if chain_id == ' ':
                    chain_id = 'A' # Andriy did not put chain_id in the target pdb file
                res_num = line[22:26].strip()
                if (chain_id, res_num) not in mapping:
                    mapping.append((chain_id, res_num))
                    seq += resname_one
                    resnum_to_res[res_num] = resname_one

    # Check for missing residues by examining if residue numbers are continuous
    # Get all residue numbers from the mapping
    res_nums = [int(res_num) for chain_id, res_num in mapping if res_num.isdigit()]
    if res_nums:
        min_res, max_res = min(res_nums), max(res_nums)
        expected_range = set(range(min_res, max_res + 1))
        actual_set = set(res_nums)
        missing_residues = expected_range - actual_set
        if missing_residues:
            print(f"Warning: {pdb_file} Missing residues detected: {sorted(missing_residues)}")
    
    assert len(seq) == len(mapping) == len(resnum_to_res)
    return seq, resnum_to_res, target_lines

def find_target_seq_in_massivefold(massivefold_pdb, seq, resnum_to_res:dict, output_dir='/scratch/'):
    chain_lines = {}
    chain_match_count = {}
    chain_match_percentage = {}
    all_match_chain_lines = []
    with open(massivefold_pdb, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                resname_three = line[17:20].strip()
                resname_one = three_to_one[resname_three]
                chain_id = line[21]
                if chain_id == ' ':
                    chain_id = 'A'
                res_num = line[22:26].strip()
                if res_num in resnum_to_res:
                    if resnum_to_res[res_num] == resname_one:
                        if chain_id not in chain_lines:
                            chain_lines[chain_id] = []
                        chain_lines[chain_id].append(line)
                        if chain_id not in chain_match_count:
                            chain_match_count[chain_id] = {}
                        if res_num not in chain_match_count[chain_id]:
                            chain_match_count[chain_id][res_num] = 1
                    else:
                        if chain_id not in chain_lines:
                            chain_lines[chain_id] = []
                        chain_lines[chain_id].append(line)
                        if chain_id not in chain_match_count:
                            chain_match_count[chain_id] = {}
                        if res_num not in chain_match_count[chain_id]:
                            chain_match_count[chain_id][res_num] = 0
    
    # normalize the chain_match_count by the length of the seq
    for chain_id in chain_match_count:
        total_matches = sum(chain_match_count[chain_id].values())
        chain_match_percentage[chain_id] = total_matches / len(seq)

    # Find the best chain
    # if no chain_match_percentage is greater than 0.99, print the massivefold_pdb
    # get the chain with the highest chain_match_percentage, if there are multiple just take the first one
    if all(value < 0.99 for value in chain_match_percentage.values()):
        print(f'{massivefold_pdb} has no chain_match_percentage greater than 0.95')
        sys.exit(1)
    elif any(value > 1.01 for value in chain_match_percentage.values()):
        print(f'{massivefold_pdb} has chain_match_percentage greater than 1.01, please check for problems')
        sys.exit(1)
    
    best_chain = max(chain_match_percentage.items(), key=lambda x: x[1])[0]
    best_match_chain_lines = chain_lines[best_chain]
    chain_begin = min(int(res_num) for res_num in resnum_to_res.keys())
    chain_end = max(int(res_num) for res_num in resnum_to_res.keys())
    with open(massivefold_pdb, 'r') as f_massivefold:
        for line in f_massivefold:
            if line.startswith('ATOM'):
                chain_id = line[21]
                if chain_id == best_chain:
                    if int(line[22:26].strip()) >= chain_begin and int(line[22:26].strip()) <= chain_end:
                        all_match_chain_lines.append(line)
    return chain_match_percentage, best_match_chain_lines, best_chain, chain_begin, chain_end, all_match_chain_lines


def make_lga_inputs(target):
    subunit = None
    if target.startswith('T1271') or target.startswith('T1272'):
        id = target[1:7]
        domain = target[8:10]
        if not domain.startswith('D'):
            print(f'{target} incorrect domain')
            sys.exit(1)
    else:
        id = target[1:5]
        domain = target[6:8]
        if not domain.startswith('D'):
            subunit = target[5:7]
            domain = target[8:10]
            if not domain.startswith('D'):
                print(f'{target} incorrect domain, please check')
                sys.exit(1)
    
    print(f'{target} domain: {domain}')
    T_folder = 'T'+id
    H_folder = 'H'+id
    target_path = os.path.join(target_dir, target)
    # check if either folder exists under massivefold_dir
    if os.path.exists(os.path.join(massivefold_dir, T_folder)) or os.path.exists(os.path.join(massivefold_dir, H_folder)):
        massivefold_home_dir = os.path.join(massivefold_dir, T_folder) if os.path.exists(os.path.join(massivefold_dir, T_folder)) else os.path.join(massivefold_dir, H_folder)
        # check if these 8 directories exist under path
        # afm_basic
        # afm_dropout_full
        # afm_dropout_full_woTemplates
        # afm_dropout_full_woTemplates_r3
        # afm_dropout_noSM_woTemplates
        # afm_woTemplates
        # cf_dropout_full_woTemplates
        # cf_woTemplates
        sub_dirs = ['afm_basic', 'afm_dropout_full', 'afm_dropout_full_woTemplates', 'afm_dropout_full_woTemplates_r3', 
                    'afm_dropout_noSM_woTemplates', 'afm_woTemplates', 'cf_dropout_full_woTemplates', 'cf_woTemplates']
        for sub_dir in sub_dirs:
            if not os.path.exists(os.path.join(massivefold_home_dir, sub_dir)):
                print(f'{sub_dir} does not exist under {massivefold_home_dir}')
                sys.exit(1)

        target_seq, resnum_to_res, target_lines = get_target_seq_and_mapping(target_path)
        for sub_dir in sub_dirs:
            massivefold_sub_dir = os.path.join(massivefold_home_dir, sub_dir)
            massivefold_pdb_files = [file for file in os.listdir(massivefold_sub_dir) if file.endswith('.pdb')]
            massivefold_pdb_files.sort()
            for massivefold_pdb_file in massivefold_pdb_files:
                massivefold_pdb_file_path = os.path.join(massivefold_sub_dir, massivefold_pdb_file)
                chain_match_percentage, best_match_chain_lines, best_chain, chain_begin, chain_end, all_match_chain_lines = find_target_seq_in_massivefold(massivefold_pdb_file_path, target_seq, resnum_to_res)
                if subunit is not None:
                    output_dir = os.path.join('/scratch/', 'T'+massivefold_sub_dir.split('/')[-2][1:]+subunit+'-'+domain, sub_dir)
                else:
                    output_dir = os.path.join('/scratch/', 'T'+massivefold_sub_dir.split('/')[-2][1:]+'-'+domain, sub_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(os.path.join(output_dir, 'MOL2')):    
                    os.makedirs(os.path.join(output_dir, 'MOL2'))
                if not os.path.exists(os.path.join(output_dir, 'TMP')):
                    os.makedirs(os.path.join(output_dir, 'TMP'))
                if not os.path.exists(os.path.join(output_dir, 'RESULTS')):
                    os.makedirs(os.path.join(output_dir, 'RESULTS'))
                with open(os.path.join(output_dir+'/MOL2', f'{massivefold_pdb_file.split(".")[0]}'), 'w') as f:
                    f.write('MOLECULE\t'+massivefold_pdb_file.split(".")[0]+'\n')
                    for line in best_match_chain_lines:
                        f.write(line)
                    f.write('END\n')
                    f.write('MOLECULE\t'+target.split('.')[0]+'\n')
                    for line in target_lines:
                        # the last character of the line cannot be 'H', which means H atoms are not included
                        if line[-2] != 'H':
                            f.write(line[:21]+best_chain+line[22:])
                    f.write('END\n')
                # break
            # break
        # break
    else:
        print(f'{T_folder} or {H_folder} does not exist under {massivefold_dir}')
        sys.exit(1)
    # remove the last layer of the output_dir
    output_dir = output_dir.rstrip('/')
    output_dir = output_dir[:output_dir.rfind('/')]
    return output_dir

def run_lga(lga_dir):
    lga_input_dir = os.path.join(lga_dir, 'MOL2')
    lga_files = [file for file in os.listdir(lga_input_dir)]
    print(lga_files)
    lga_files.sort()
    os.chdir(lga_dir)
    with open(os.path.join(lga_dir, 'lga_cmd'), 'w') as f:
        f.write('ulimit -s unlimited\n')
        for lga_file in lga_files:
            f.write(f'/data/data1/conglab/jzhan6/software/LGA_package_src/lga -3  -ie  -o1  -sda  -d:4  -gdc_sc {lga_file}\n')
    os.system('bash lga_cmd')

def move_lga_results(lga_dir, home_node_dir="/home2/s439906/project/CASP16/monomer/lga_output"):
    lga_tmp_dir = os.path.join(lga_dir, 'TMP')
    # mv all .lga files under lga_tmp_dir to home_node_dir
    lga_out_dir = os.path.join(home_node_dir, lga_dir.split('/')[-2], lga_dir.split('/')[-1])
    if not os.path.exists(lga_out_dir):
        os.makedirs(lga_out_dir)
    for lga_file in os.listdir(lga_tmp_dir):
        if lga_file.endswith('.lga'):
            os.system(f'mv {lga_tmp_dir}/{lga_file} {lga_out_dir}')


output_dir = make_lga_inputs(target)
sub_dirs = ['afm_basic', 'afm_dropout_full', 'afm_dropout_full_woTemplates', 'afm_dropout_full_woTemplates_r3', 
                        'afm_dropout_noSM_woTemplates', 'afm_woTemplates', 'cf_dropout_full_woTemplates', 'cf_woTemplates']

lga_dirs = [os.path.join(output_dir, sub_dir) for sub_dir in sub_dirs]
with Pool(processes=10) as pool:
    pool.map(run_lga, lga_dirs)

with Pool(processes=10) as pool:
    pool.map(move_lga_results, lga_dirs)


# run_lga('/scratch/T1201-D1/')
# output_dir = make_lga_inputs(targets[0])
# run_lga(output_dir)
# move_lga_results(output_dir)