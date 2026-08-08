"""
Step 2 — Turn per-subject ROI time series into a functional connectivity
matrix (Pearson correlation between every pair of brain regions), then
flatten the upper triangle into one feature vector per subject.

Input : {OUTPUT_DIR}/{subject}/{subject}_roi_time_series.csv
Output: {OUTPUT_DIR}/{subject}/{subject}_connectivity.csv   (116x116 per subject)
        {FEATURE_MATRIX_PATH}                                (55 x 6670 final dataset)
"""

import os
import numpy as np
import pandas as pd

from config import SUBJECTS, OUTPUT_DIR, FEATURE_MATRIX_PATH, PARTICIPANTS_TSV


def compute_connectivity_matrices() -> None:
    for subject in SUBJECTS:
        subject_dir = os.path.join(OUTPUT_DIR, subject)
        roi_csv = os.path.join(subject_dir, f"{subject}_roi_time_series.csv")
        out_csv = os.path.join(subject_dir, f"{subject}_connectivity.csv")

        if not os.path.exists(roi_csv):
            print(f"  [skip] {subject}: no ROI time series found")
            continue

        roi_df = pd.read_csv(roi_csv)
        # Symmetric connectivity matrix: correlation of every region's
        # BOLD time series against every other region's
        conn_matrix = np.corrcoef(roi_df.values, rowvar=False)

        conn_df = pd.DataFrame(conn_matrix, index=roi_df.columns, columns=roi_df.columns)
        conn_df.to_csv(out_csv, index=True)
        print(f"  [ok] {subject}: connectivity {conn_df.shape}")


def build_feature_matrix() -> pd.DataFrame:
    """Flatten the upper triangle (excluding diagonal) of each subject's
    connectivity matrix into a single feature vector, since the matrix
    is symmetric and the diagonal is always 1."""
    features, subject_list = [], []

    for subject in SUBJECTS:
        conn_path = os.path.join(OUTPUT_DIR, subject, f"{subject}_connectivity.csv")
        if not os.path.exists(conn_path):
            print(f"  [skip] {subject}: no connectivity matrix found")
            continue

        conn_matrix = pd.read_csv(conn_path, index_col=0).values
        upper_triangle = conn_matrix[np.triu_indices_from(conn_matrix, k=1)]
        features.append(upper_triangle)
        subject_list.append(subject)

    X = np.array(features)
    print(f"Feature matrix shape: {X.shape}  (subjects x connectivity features)")

    df = pd.DataFrame(X)
    df.insert(0, "Subject", subject_list)
    df.to_csv(FEATURE_MATRIX_PATH, index=False)
    print(f"Saved: {FEATURE_MATRIX_PATH}")
    return df


def build_labels(subject_list: list[str]) -> np.ndarray:
    """Map OpenNeuro participant groups to a binary PD vs. healthy-control label."""
    participants_df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
    label_map = {"Control": 0, "PD-NC": 1, "PD-MCI": 1}

    y = np.array([
        label_map[participants_df.loc[participants_df["participant_id"] == s, "group"].values[0]]
        for s in subject_list
    ])
    print(f"Labels: {len(y)} total  |  HC={np.sum(y == 0)}  PD={np.sum(y == 1)}")
    return y


if __name__ == "__main__":
    compute_connectivity_matrices()
    feature_df = build_feature_matrix()
    build_labels(feature_df["Subject"].tolist())
