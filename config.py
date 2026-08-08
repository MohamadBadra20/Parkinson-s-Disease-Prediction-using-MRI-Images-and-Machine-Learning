"""
Central configuration for the Parkinson's disease fMRI classification pipeline.

Edit the paths below to match your local setup. Everything else in the
pipeline reads from here so there are no hardcoded paths in the scripts.

Data source (public, no access request needed):
    https://openneuro.org/datasets/ds005892/versions/1.0.0
"""

import os

# Root folder where you downloaded/derived everything for this project
PROJECT_ROOT = "C:/Users/XPRISTO/Desktop/PD_PROJECT"

# fMRIPrep output directory (one subfolder per subject, BIDS derivatives layout)
FMRIPREP_DIR = os.path.join(PROJECT_ROOT, "fmriprep_out")

# Raw AAL-116 atlas file (SPM12 AAL atlas, downloaded separately)
AAL_ATLAS_PATH = os.path.join(PROJECT_ROOT,"aal_for_SPM12", "aal_for_SPM12", "atlas", "AAL.nii")

# Where intermediate outputs (resampled atlas, per-subject ROI time series,
# connectivity matrices, final feature matrix) get written
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "derivatives")

# OpenNeuro participants.tsv (subject_id, group, age, sex)
PARTICIPANTS_TSV = os.path.join(PROJECT_ROOT, "participants.tsv")

RESAMPLED_ATLAS_PATH = os.path.join(OUTPUT_DIR, "AAL_resampled.nii.gz")
FEATURE_MATRIX_PATH = os.path.join(OUTPUT_DIR, "feature_matrix.csv")

# Where trained models / results and generated figures are saved
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
ALL_RESULTS_PKL = os.path.join(MODELS_DIR, "all_results.pkl")

RANDOM_STATE = 42

SUBJECTS = [
    "sub-MJF001", "sub-MJF002", "sub-MJF003", "sub-MJF006", "sub-MJF007",
    "sub-MJF008", "sub-MJF009", "sub-MJF010", "sub-MJF011", "sub-MJF012",
    "sub-MJF013", "sub-MJF014", "sub-MJF015", "sub-MJF016", "sub-MJF017",
    "sub-MJF019", "sub-MJF020", "sub-MJF024", "sub-MJF026", "sub-MJF027",
    "sub-MJF028", "sub-MJF029", "sub-MJF030", "sub-MJF031", "sub-MJF032",
    "sub-MJF033", "sub-MJF034", "sub-MJF035", "sub-MJF036", "sub-MJF037",
    "sub-MJF038", "sub-MJF039", "sub-MJF040", "sub-MJF041", "sub-MJF042",
    "sub-MJF043", "sub-MJF044", "sub-MJF045", "sub-MJF046", "sub-MJF049",
    "sub-MJF050", "sub-MJF051", "sub-MJF052", "sub-MJF054", "sub-MJF056",
    "sub-MJF057", "sub-MJF062", "sub-MJF063", "sub-MJF068", "sub-MJF069",
    "sub-MJF072", "sub-MJF077", "sub-MJF079", "sub-MJF080", "sub-MJF082",
]
