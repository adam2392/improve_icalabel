## Labeling Components Workflow

The goal: label ICA components into one of seven labels:

- "brain"
- "heart beat"
- "eye blink"
- "muscle artifact"
- "channel noise"
- "line noise"
- "other"

where the main priority is determining if any of the component is part of a "brain" signal.

Within the GUI that ``mne-icalabel`` will have (https://github.com/mne-tools/mne-icalabel/pull/66), to label the components of each dataset, we'll follow these steps:

- open up BIDS file corresponding to the ICA + the Raw/Epochs instance (at most 1 minute lengths)
- show the plot_properties plots with 1 minute of the time-series by default
- label components with color-coded labels
- closing the window will now save back to the BIDs files.
- update script to do the next file

For full details, see `labeling_workflow.md` file.

## Preliminary Workflow to QA Annotater

For any new researcher annotating the ICA components, one should QA their choices on a set of sample datasets. For now, we will use this dataset: https://openneuro.org/datasets/ds004132/download

1. Download the dataset
2. Run ICA script for ``sub-1``: `mnestudy/ica_to_bids/ants.py`
3. Run `examples/plot_ica_components.ipynb` to plot the features for each component
4. modify the `deriatives/ica/average/sub-1/**/*_channels.tsv` to change `status` to good/bad and then `status_description` to one of the component labels.