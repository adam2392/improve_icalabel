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

# Labeling Raw dataset

Here, we would like you to develop an understanding of what is good/bad EEG data so you are more comfortable working with neuroscience data. EEG data is generally "subjective" because "good" vs "bad" signals are up to the researcher. However, there are some stereotypical signatures one can look for that so that you can quickly annotate your data.

The Raw dataset for the ``ANTs`` dataset currently still needs to be QAed. The files with the ``*proc-pp*`` in filename correspond to the EEG files that have been preprocesed with an automated pipeline, known as [PyPrep](https://github.com/sappelhoff/pyprep). Your first task would be to double check the output of this. There are two MNE-Python tutorials online that walk through a description of "bad channels" and "bridge electrodes". It will be educational and beneficial for you to walk through these yourself using your installed MNE-Python and other relevant packages needed to run the code:

- https://mne.tools/stable/auto_tutorials/preprocessing/15_handling_bad_channels.html
- https://mne.tools/dev/auto_examples/preprocessing/eeg_bridging.html

Note: If you need to install MNE-Python, https://mne.tools/dev/install/manual_install.html has a great walk-through.

Walk through the following subjects: ``sub-1`` and ``sub-2``. You will read in the files using [mne-bids](). You can use the following script:

The above linked tutorials will then walk you through how to "annotate" or "mark" bad channels, or how they will appear on your plotter. What I would like you to do is the following:

- familiarize yourself with the code/concepts presented in the two tutorials and play around with it to see if you understand conceptually what is happening.
- double check the bad/good annotations that are present in the dataset for `sub-1` and `sub-2`. You can see what are bad/good electrodes in the `sub-1/eeg/sub-1_task-neurofeedback_proc-pp_channels.tsv` file for example. This is what `read_raw_bids()` is reading data from underneath the hood.
- have some screenshots of "good" vs "bad" electrodes from the visualization output based on your impression, similar to what you see in the linked tutorials. There is no right answer. This is just for us to QA your understanding of reading good/bad EEG.

After you are more sure of your ability to qualitatively check the EEG traces, we'll ask you to go through the rest of the ANT dataset for all subjects with ``*proc-pp*`` files.