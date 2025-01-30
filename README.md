# Critical Assessment of Structure Prediction Round XVI (CASP16)

This is the codebase for CASP16 monomer and oligomer assessment.

For monomer I am using my pipeline. For oligomers I am directly taking the results from Qian for convenience because there are complications in the methodology. Both are independently validated by each of us.

## Monomer

`data_process.py` is the main script for processing the data. It reads the CASP16 data and generates the input files for ranking related scripts.

`get_raw_scores.py` reads the input files and generates the raw scores for each model, which will further be used for analysis of performances.

`sum_z_scores.py` reads the z-scores and gets sum of z-scores for each measure. This will be used for plotting ranks.

`monomer_EU_info` is the metadata for the monomer EU data. It contains the information of each EU.

`step7_results` is the file for difficulty partitioning.

the jupyter notebooks are used for plotting the results.


## Oligomer