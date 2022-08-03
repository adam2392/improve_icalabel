# Improvements to ICA Labeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![status](https://joss.theoj.org/papers/d91770e35a985ecda4f2e1f124977207/status.svg)](https://joss.theoj.org/papers/d91770e35a985ecda4f2e1f124977207)

This repository is a research repository for improving the automatic labeling of ICA components. We will be leveraging the MNE-ICALabel package, which is currently in review at JOSS (see badge).

# Code Layout

Each subdirectory of `mnestudy` will house some scripts to perform processing and analysis.

- ``raw_to_bids/``: For datasets with existing Raw data, we will need to preprocess them, convert to BIDS, and then run ICA and convert that to BIDS. For ICA portion, scripts should be housed in ``ica_to_bids/``.
- ``ica_to_bids/``: This converts existing ICA data to BIDS format. It will optionally also run ICA analysis. Some datasets will only contain the ICA portion.

You'll need to install mne-icalabel:

    pip install https://api.github.com/repos/mne-tools/mne-icalabel/zipball/main

# Data Layout

We would ideally like the Raw data saved in one single file (split up if the format requires it). The Raw data should be saved according to [BIDS-EEG](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html).

## ICA Directory Layout

When we perform ICA, we would like to split the signal up into non-overlapping Epochs of at most 1 minute long. See [discussion on GH](https://github.com/mne-tools/mne-icalabel/issues/12). Then they will be saved according to a non-compliant BIDS-derivative format.

```
root/
    derivatives/
        ica/
            dataset_description.json
            sub-001_<other bids identifiers>_run-001_desc-ica_ieeg.h5
            sub-001_<other bids identifiers>_run-001_desc-ica_ieeg.json
            sub-001_<other bids identifiers>_run-002_desc-ica_ieeg.h5
            sub-001_<other bids identifiers>_run-002_desc-ica_ieeg.json
            ...
            
```

In `sub-001`, there are multiple ICA files mapped to the exact same raw file, meaning the raw file was more than 1 minute long in recording. Thus we should save multiple ICA outputs, which each correspond to a "different dataset" with possibly different component labels. The reason we separate this out is to capture possibly artifacts present in only a subset of the recording, while still having enough data to estimate the ICA.

## Proposed Metadata

There should be the same type of processing done for each ICA dataset. The ``dataset_description.json`` file should store the parametrizations of each pipeline. For example, MNE processing, pyprep processing, etc.

# Proposed Preprocessing

## Raw -> ICA

This is the tentative workflow to go from Raw data to ICA.

1. Filtering: bandpass filter from 1-100 Hz with FIR with "zero-double" phase, hamming window and padding the edges.
2. Montage: Unless otherwise specified, it most likely is standard 1020.
3. Detection of Bad Channels with PyPrep: Use default settings(?)
4. Looking for Bridged Electrodes:
5. Referencing: Since ICLabel worked with common average, we'll just do that too.
6. ICA: run ICA with the ``picard`` method.
7. save ICA to disc.

## Labeling Components Workflow

For full details, see `labeling_workflow.md` file.