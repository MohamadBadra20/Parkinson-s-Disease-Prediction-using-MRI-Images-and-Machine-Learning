# fmriprep_out/

This folder is intentionally empty in the repository (see `.gitignore` —
`fmriprep_out/` is excluded because it's regeneratable and far too large for
git, typically tens of GB for this dataset).

## What belongs here

The fMRIPrep-preprocessed output for all 55 subjects, in BIDS derivatives
layout — one subfolder per subject:

```
fmriprep_out/
├── sub-MJF001/
│   ├── anat/
│   │   └── sub-MJF001_desc-preproc_T1w.nii.gz
│   └── func/
│       └── sub-MJF001_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
├── sub-MJF002/
│   └── ...
└── ... (one folder per subject in config.SUBJECTS)
```

`notebooks/01_atlas_resampling_and_roi_extraction.ipynb` and
`src/01_extract_roi_timeseries.py` both read from here — specifically the
`*_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` file per
subject (the MNI152-normalized, motion-corrected resting-state BOLD scan).

## How to regenerate it

1. **Get the raw data** — the 55-subject resting-state fMRI dataset is public
   on OpenNeuro: https://openneuro.org/datasets/ds005892/versions/1.0.0

2. **Run fMRIPrep via Docker** (requires Docker installed):
   ```bash
   docker run -ti --rm \
     -v /path/to/raw_bids_data:/data:ro \
     -v /path/to/fmriprep_out:/out \
     -v /path/to/freesurfer_license.txt:/opt/freesurfer/license.txt \
     nipreps/fmriprep:latest \
     /data /out participant \
     --output-spaces MNI152NLin2009cAsym \
     --fs-license-file /opt/freesurfer/license.txt
   ```
   See the official docs for the exact flags and a FreeSurfer license (free,
   required, get one at https://surfer.nmr.mgh.harvard.edu/registration.html):
   https://fmriprep.org/en/stable/usage.html

3. **Point `config.py` at the output** — `FMRIPREP_DIR` should resolve to
   this folder once it's populated.

4. **Continue the pipeline** — run `notebooks/01_atlas_resampling_and_roi_extraction.ipynb`
   onward once this folder has real subject data.

## Note

fMRIPrep is compute- and memory-intensive — expect several hours per subject
depending on hardware. The full 55-subject run that produced this project's
results was done incrementally, not in a single batch.
