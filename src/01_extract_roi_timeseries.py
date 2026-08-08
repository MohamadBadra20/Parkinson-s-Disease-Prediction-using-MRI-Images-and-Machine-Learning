"""
Step 1 — Resample the AAL-116 atlas to native fMRI space, then extract
one BOLD time series per brain region for every subject.

Prerequisite: fMRIPrep has already been run (see README) so that each
subject has a preprocessed, MNI152-normalized functional image.

Input : {FMRIPREP_DIR}/{subject}/func/*_desc-preproc_bold.nii.gz
Output: {OUTPUT_DIR}/{subject}/{subject}_roi_time_series.csv  (one per subject)
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img

from config import SUBJECTS, FMRIPREP_DIR, AAL_ATLAS_PATH, OUTPUT_DIR, RESAMPLED_ATLAS_PATH


def get_fmri_path(subject: str) -> str:
    return os.path.join(
        FMRIPREP_DIR, subject, "func",
        f"{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )


def resample_atlas_to_reference(reference_subject: str = SUBJECTS[0]) -> tuple[np.ndarray, np.ndarray]:
    """Resample the AAL atlas onto one subject's fMRI grid (nearest-neighbor,
    since atlas voxels are integer region labels and must not be interpolated)."""
    fmri_img = nib.load(get_fmri_path(reference_subject))
    atlas_img = nib.load(AAL_ATLAS_PATH)

    atlas_resampled_img = resample_to_img(
        source_img=atlas_img, target_img=fmri_img, interpolation="nearest"
    )
    atlas_data = atlas_resampled_img.get_fdata()

    region_ids = np.unique(atlas_data)
    region_ids = region_ids[region_ids != 0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nib.save(atlas_resampled_img, RESAMPLED_ATLAS_PATH)
    print(f"Resampled atlas saved to {RESAMPLED_ATLAS_PATH}")
    print(f"Regions found: {len(region_ids)}  (expected 116 for AAL)")

    return atlas_data, region_ids


def extract_all_subjects(atlas_data: np.ndarray, region_ids: np.ndarray) -> None:
    for subject in SUBJECTS:
        fmri_path = get_fmri_path(subject)
        if not os.path.exists(fmri_path):
            print(f"  [skip] {subject}: fMRI file not found")
            continue

        fmri_data = nib.load(fmri_path).get_fdata()

        roi_time_series = []
        for region in region_ids:
            mask = atlas_data == region
            mean_signal = fmri_data[mask].mean(axis=0)
            roi_time_series.append(mean_signal)

        roi_time_series = np.array(roi_time_series).T  # shape: (n_timepoints, n_regions)

        subject_dir = os.path.join(OUTPUT_DIR, subject)
        os.makedirs(subject_dir, exist_ok=True)
        out_path = os.path.join(subject_dir, f"{subject}_roi_time_series.csv")

        df = pd.DataFrame(
            roi_time_series,
            columns=[f"ROI_{int(r)}" for r in region_ids],
        )
        df.to_csv(out_path, index=False)
        print(f"  [ok] {subject}: {df.shape}")


if __name__ == "__main__":
    atlas_data, region_ids = resample_atlas_to_reference()
    extract_all_subjects(atlas_data, region_ids)
