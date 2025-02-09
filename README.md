# CASP16 (Critical Assessment of Structure Prediction Round XVI)

This is the codebase for CASP16 monomer and oligomer assessment.

Please check each directory for source code to rank groups and plot results. Note that the pipeline for each assessment is rather different.

For monomer I am using my pipeline. For oligomers I am directly taking the results from Qian for convenience because there are complications in the methodology. Both are independently validated by each of us.

**Raw scores and processed scores are available upon request.**

## Monomer

`data_process.py` is the main script for processing the data. It reads the CASP16 data and generates the input files for ranking related scripts.

`get_raw_scores.py` reads the input files and generates the raw scores for each model, which will further be used for analysis of performances.

`sum_z_scores.py` reads the z-scores and gets sum of z-scores for each measure. This will be used for plotting ranks.
`ranking.ipynb` and `head_to_head_playground.ipynb` are the scripts for ranking the models and generating the head-to-head test results.

The above scripts will provide the input for downstream analysis.
Other jupyter notebooks are used for plotting the results.

`CASP15_data_job` is the job for processing the data from CASP15.

`monomer_EU_info` is the metadata for the monomer EU data. It contains the information of each EU.

`step7_results` is the file for difficulty partitioning.



## Oligomer

`data_process.py` is the script for processing the data from CASP16 website. However, since we have our own pipeline, the official data is not used for the final results.

`get_raw_scores.ipynb` is the script for getting the raw scores from Qian's pipeline. These raw data will be further used for analysis of performances.

`rank_and_h2h_from_Qian.ipynb` is the script for ranking the models using Qian's data and generating the head-to-head test results.

The above scripts will provide the input for downstream analysis.
Other jupyter notebooks are used for plotting the results.

`process_data_CASP15.py` processes the data from CASP15.

`stoich_bg_distribution.csv` is the file for background distribution of stoichiometry in PDB.

`target_weights` is the file for how many subunits in each target. They will be transformed into weights for each target under our pipeline.


## Qian's data

Some of the data are directly from Qian's pipeline. The notation is as follows:

stage1: raw data process

stage2: produce rankings

stage3: get insights from the data
