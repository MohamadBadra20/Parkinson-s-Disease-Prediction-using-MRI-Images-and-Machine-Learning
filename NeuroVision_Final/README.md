# NeuroVision — Parkinson's fMRI Research Web Application

This folder is the deployable Flask web application for the Parkinson's disease
resting-state fMRI classification project.

## Integrated model package

`soft_voting_deployment/` is already populated from the project's trained
`models/all_results.pkl` file:

- SVM_Linear
- LogisticRegression
- RidgeClassifier
- AAL.nii (116-region atlas)
- config.pkl

The application extracts 6,670 upper-triangle Pearson connectivity features,
passes them through the three saved pipelines, averages their P(PD)
probabilities, and applies the project's fixed 0.35 soft-voting threshold.

## Important compatibility

The saved scikit-learn pipelines were trained with scikit-learn 1.8.0.
Install the versions in `requirements.txt` rather than upgrading scikit-learn
to an unrelated major/minor version.

## Run on Windows

From this folder:

```text
conda create -n neurovision python=3.10 -y
conda activate neurovision
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/`.

## Project structure

```text
NeuroVision_Final/
├── app.py
├── database.db
├── requirements.txt
├── templates/
├── Images/
├── uploads/
├── previews/
└── soft_voting_deployment/
    ├── config.pkl
    ├── AAL.nii
    ├── SVM_Linear.pkl
    ├── LogisticRegression.pkl
    └── RidgeClassifier.pkl
```

## Input requirement

The upload must be a 4D NIfTI fMRI image (`.nii` or `.nii.gz`). The atlas is
resampled to the uploaded image using nearest-neighbor interpolation. The
application expects 116 AAL regions and produces 6,670 connectivity features.

This is a research prototype, not a medical diagnostic device. Predictions
must not be used as a substitute for clinical assessment.
