import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne.io import BaseRaw
from mne.preprocessing import ICA, compute_bridged_electrodes
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids


def preprocess_raw(raw: BaseRaw, bads=[]):
    # ensuure that Raw data is loaded onto RAM
    # MNE functions generally require this
    raw.load_data()

    # set bad channels here
    raw.info["bads"] = bads

    # set montage - EEG locations from a template
    raw.set_montage("standard_1020")

    # upper/lower-cutoff of frequencies
    h_freq = 100.0
    l_freq = 1.0
    line_freq = raw.info["line_freq"]  # 60 in USA, 50 in EUrope

    # now, we will attempt to compute the bridged EEG electrodes
    # See: https://mne.tools/dev/auto_examples/preprocessing/eeg_bridging.html#sphx-glr-auto-examples-preprocessing-eeg-bridging-py
    bridged_idx, ed_matrix = compute_bridged_electrodes(raw)

    # for each subject, we want to interpolate a different set of bridged electrodes possibly
    # TODO: we can refactor this code later and re-run the analysis when `interpolate_bridged_electrodes`
    # function accepts sets bridged indices of size greater than 2. Right now, it only interpolates
    # two pairs of electrodes.

    # now let us look at the bridged electrodes
    unbridged_raw = mne.preprocessing.interpolate_bridged_electrodes(
        raw.copy(), bridged_idx=bridged_idx
    )

    # show the unbridged filtered version
    unbridged_filt_raw = unbridged_raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    unbridged_filt_raw = unbridged_filt_raw.notch_filter(freqs=line_freq)

    # also show the Raw EEG voltage traces before and after bridging-analysis/filtering
    raw.plot(n_channels=len(raw.ch_names), duration=20, scalings=dict(eeg=2e-4))
    unbridged_filt_raw.plot(
        n_channels=len(raw.ch_names), duration=20, scalings=dict(eeg=2e-4)
    )

    # now let us examine the bridged electrodes and note them down
    fig = mne.viz.plot_bridged_electrodes(
        raw.info,
        bridged_idx=bridged_idx,
        ed_matrix=ed_matrix,
        title="Bridging Elecs Topographic Map Analysis",
        topomap_args=dict(show=False),
    )
    plt.show(block=True)

    # now show the figure but block program execution...
    # Note: currently there is no `block` parameter in the plotting bridged
    # electrodes function, so we have to do this instead. In future, sub this out
    # and add the kwarg to the function itself.
    # fig.show(block=True)
    return unbridged_filt_raw


def ica_analysis(raw, n_components=0.99, plot=True):
    # ICA FILTERING with n_components being around the number of channels there are
    ica = ICA(n_components=n_components, max_iter="auto", random_state=97)
    ica.fit(raw)

    if plot:
        # plot ICA
        fig = ica.plot_properties(raw)
        plt.show(block=True)
    return ica


def annotate_ica_components(raw, ica, fname):
    from mne_icalabel.annotation import write_components_tsv
    from mne_icalabel.gui import label_ica_components

    # open up the GUI
    gui = label_ica_components(raw, ica, show=True, block=True)

    # the ICA instance will be modified in-place. So the labels of the
    # ICA components will be set in the 'labels_' property
    print(ica.labels_)

    # To save this to disc,
    write_components_tsv(ica, fname)


def main():
    root = "~/Dropbox/ds004178-download/"  # change to wherever the dataset is housed for you
    deriv_root = Path(root) / "derivatives" / "preprocessed"
    subject = "3"  # change this for each subject
    task = "neurofeedback"
    proc = "pp"
    extension = ".vhdr"

    bids_path = BIDSPath(
        root=root, subject=subject, task=task, processing=proc, extension=extension
    )

    # initiate the set of bad channels
    # TODO: Aaron: you should fill in here to explicitly pass the list of bad channels
    # per subject
    bads =[]
    if subject == '0':
        bads = []
    elif subject == '1':
        bads = []
    

    # this is now a MNE-Python raw object
    raw = read_raw_bids(bids_path)

    # preprocess the Raw file
    preproc_raw = preprocess_raw(raw, bads)

    # write now to BIDS
    pproc_bids_path = BIDSPath(
        subject=subject,
        task=task,
        processing="pproc",
        extension=".edf",
        root=deriv_root,
    )
    # preproc_raw = interpolate_bridged_electrodes(..., <add the electrodes to interpolate manually per subject>)
    write_raw_bids(preproc_raw, pproc_bids_path, format="EDF", allow_preload=True, overwrite=True)


def main_ica():
    root = "~/Dropbox/ds004178-download/"  # change to wherever the dataset is housed for you
    deriv_root = Path(root) / "derivatives" / "preprocessed"
    subject = "3"  # change this for each subject
    task = "neurofeedback"
    proc = "pp"
    extension = ".vhdr"

    pproc_bids_path = BIDSPath(
        subject=subject,
        task=task,
        processing="pproc",
        extension=".edf",
        root=deriv_root,
    )
    raw = read_raw_bids(pproc_bids_path)

    # re-reference the data to common average reference (average signal)
    preproc_raw = preproc_raw.set_eeg_reference("average")

    # run ICA
    ica = ica_analysis(preproc_raw, n_components=0.99)

    # save the ICA to the same directory
    fname = pproc_bids_path.copy().update(
        extension=".fif.gz", processing="ica", check=False, suffix="ica"
    )
    print(f"Saving ICA instance to disc at: {fname}")
    ica.save(fname)

    # annotate the components
    fname = pproc_bids_path.copy().update(
        extension=".tsv",
        processing="annotations",
        check=False,
        suffix="deriv-aaron_ica",
    )
    print(f"Saving annotation labels to disc at: {fname}")
    annotate_ica_components(raw, ica, fname)


if __name__ == "__main__":
    main()
    # main_ica()
