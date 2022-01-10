# MIMIC-III ICU Mortality Prediction

Mortality prediction for ICU records with length-of-stay between 1 day and 2 days(1 <= (ICU Stay LOS) <= 2). Chart Events information of first three hours since ICU in-time is used for prediction.

## Dataset
Mimic-III dataset can only be accessed through CITI training from https://mimic.physionet.org/gettingstarted/access/. The data is not provided in this repository. Download the CSVs, and modify the base_path location in mimic3_icu_mortality/utils/load_prep_csv.py.

## Structure
The content of the repository consists of four parts:
1. Loading and restoring conditional data from ICUSTAYS.csv, ADMISSIONS.csv, CHARTEVENTS.csv, PATIENTS.csv.
2. Converting the csv created above into .npy format for Logistic Regression and RNN.
3. Conducting Logistic Regression, RETAIN GRU training and testing respectively.
4. Comparing the AUROC, AUPRC results.

The mimic3_icu_mortality/utils/load_prep_csv.py contains code for loading and restoring conditional data from the raw csv files. Creating the .npy files for Logistic Regression and RNN are done in mimic3_icu_mortality/utils/logistic_preprocessing.py, and mimic3_icu_mortality/utils/rnn_preprocessing.py respectively. Code for training and testing Logistic Regression, RNN is located in mimic3_icu_mortality/src.

## References
(1) Choi, Edward, et al. "Retain: An interpretable predictive model for healthcare using reverse time attention mechanism." arXiv preprint arXiv:1608.05745 (2016).

(2) https://github.com/mp2893/retain

(3) https://github.com/ast0414/pytorch-retain
