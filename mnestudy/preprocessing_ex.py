from mne_bids import BIDSPath, read_raw_bids


def main():
    root = '/Dropbox/ds004178-download/'  # change to wherever the dataset is housed for you
    subject= 'sub-1'  # change this for each subject
    task='neurofeedback'
    proc='pp'
    extension='.vhdr'

    bids_path = BIDSPath(root=root, subject=subject, task=task, processing=proc,
    extension=extension)

    # this is now a MNE-Python raw object
    raw = read_raw_bids(bids_path)

    # Note: you may need to install additional softwares for visualization
    # Unsure... on Windows...
    # you can plot the Raw object using its `plot()` function
    raw.plot()