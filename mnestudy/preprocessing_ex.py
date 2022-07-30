import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
from mne.io import BaseRaw
from mne.preprocessing import (ICA, compute_bridged_electrodes, corrmap,
                               create_ecg_epochs, create_eog_epochs)
from mne_bids import BIDSPath, read_raw_bids, write_raw_bids


def preprocess_raw(raw: BaseRaw):
    # ensuure that Raw data is loaded onto RAM
    # MNE functions generally require this
    raw.load_data()

    # set montage - EEG locations from a template
    raw.set_montage('standard_1020')

    # upper/lower-cutoff of frequencies
    h_freq = 100.0
    l_freq = 1.0
    line_freq = raw.info["line_freq"]  # 60 in USA, 50 in EUrope

    # now, we will attempt to compute the bridged EEG electrodes
    # See: https://mne.tools/dev/auto_examples/preprocessing/eeg_bridging.html#sphx-glr-auto-examples-preprocessing-eeg-bridging-py
    bridged_idx, ed_matrix = compute_bridged_electrodes(raw)

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
        show=False,
    )

    # now show the figure but block program execution...
    # Note: currently there is no `block` parameter in the plotting bridged
    # electrodes function, so we have to do this instead. In future, sub this out
    # and add the kwarg to the function itself.
    plt.show(block=True)

    return unbridged_filt_raw


def ica_analysis(raw):
    # ICA FILTERING with n_components being around the number of channels there are
    ica = ICA(n_components=None, max_iter="auto", random_state=97)
    ica.fit(raw)

    # plot ICA
    fig = ica.plot_properties(raw, show=False)
    fig.show(block=True)
    return ica


def manual_plot(raw, ica):
    pass


def main():
    root = "/Dropbox/ds004178-download/"  # change to wherever the dataset is housed for you
    deriv_root = Path(root) / "derivatives" / "preprocessed"
    subject = "sub-1"  # change this for each subject
    task = "neurofeedback"
    proc = "pp"
    extension = ".vhdr"

    bids_path = BIDSPath(
        root=root, subject=subject, task=task, processing=proc, extension=extension
    )

    # this is now a MNE-Python raw object
    raw = read_raw_bids(bids_path)

    # preprocess the Raw file
    preproc_raw = preprocess_raw(raw)

    # write now to BIDS
    pproc_bids_path = BIDSPath(
        subject=subject, task=task, proc="pproc", extension=".edf", root=deriv_root
    )
    write_raw_bids(preproc_raw, pproc_bids_path, format="EDF", allow_preload=True)

    # common average reference
    preproc_raw = preproc_raw.set_eeg_reference("average")

    # run ICA
    ica = ica_analysis(preproc_raw)
