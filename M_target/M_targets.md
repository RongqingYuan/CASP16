## Usage of the scripts to process M_targets.

Get interface scores: protein-NA interfaces `python step1_get_interface_score.py --type pn`. protein-protein interfaces `python step1_get_interface_score.py --type pp`.

Get target-level scores: `python step2_get_target_score.py`.

Compute z-scores: all targets `python step3_get_z_score.py`. protein-NA interfaces `python step3_get_z_score.py --type pn`. protein-protein interfaces `python step3_get_z_score.py --type pp`.

Plot ranking and heatmap: all targets `python step4_ranking_heatmap.py`. protein-NA interfaces `python step4_ranking_heatmap.py --type pn`. protein-protein interfaces `python step4_ranking_heatmap.py --type pp`.

