basic:

`python step1_process_M_score.py`


step1 for protein protein interface. protein protein interface has special cases so we use a different output_dir (some targets do not have protein protein interface)
`python step1_process_M_score.py --features prot_per_interface_qs_global prot_per_interface_ics_trimmed prot_per_interface_ips_trimmed --output_dir ./interface_score_prot_prot_v1/`



basic:

`python step2_sum_M_score_bootstrap.py`


step2 for protein protein interface: (there are special cases for protein protein interface so we use a different interface_score_path)

`python step2_sum_M_score_bootstrap.py --measures prot_per_interface_qs_global prot_per_interface_ics_trimmed prot_per_interface_ips_trimmed --interface_score_path ./interface_score_prot_prot_v1/ --output_path ./bootstrap_prot_prot_interface_score_v1/`

step2 for protein nucleotide interface:

`python step2_sum_M_score_bootstrap.py --measures prot_nucl_per_interface_qs_global prot_nucl_per_interface_ics_trimmed prot_nucl_per_interface_ips_trimmed --output_path ./bootstrap_prot_nucl_interface_score_v1/`

step7 basic:

`python step7_heatmap_interface_score.py`


step7 for protein protein interface:

`python step7_heatmap_interface_score.py --measures prot_per_interface_qs_global prot_per_interface_ics_trimmed prot_per_interface_ips_trimmed --interface_score_path ./interface_score_prot_prot_v1/  --output_path ./heatmap_prot_prot_interface/`

step7 for protein nucleotide interface:

`python step7_heatmap_interface_score.py --measures prot_nucl_per_interface_qs_global prot_nucl_per_interface_ics_trimmed prot_nucl_per_interface_ips_trimmed --output_path ./heatmap_prot_nucl_interface/`


step12 basic:
`python step12_heatmap_target_score_large.py`

step12 for protein protein interface:
`python step12_heatmap_target_score_large.py --measures prot_per_interface_qs_global prot_per_interface_ics_trimmed prot_per_interface_ips_trimmed --interface_score_path ./interface_score_large/ --output_path ./heatmap_prot_prot_interface_large/`

step12 for protein nucleotide interface:
`python step12_heatmap_target_score_large.py --measures prot_nucl_per_interface_qs_global prot_nucl_per_interface_ics_trimmed prot_nucl_per_interface_ips_trimmed --interface_score_path ./interface_score_large/ --output_path ./heatmap_prot_nucl_interface_large/`