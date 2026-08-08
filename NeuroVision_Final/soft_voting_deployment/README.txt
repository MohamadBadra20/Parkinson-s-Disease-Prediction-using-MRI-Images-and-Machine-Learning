This deployment folder is integrated with the NeuroVision Flask app.

Contains the three trained top-3 pipelines used by the project's soft-voting ensemble:
SVM_Linear, LogisticRegression, RidgeClassifier.

Threshold: 0.35
Atlas: AAL.nii
Regions: 116
Connectivity features: 6670

IMPORTANT: these serialized scikit-learn models were created with scikit-learn 1.8.0.
Use the project's requirements.txt.
