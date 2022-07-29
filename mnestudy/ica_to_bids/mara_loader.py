from typing import Tuple

import numpy as np
from mne import create_info
from mne.channels import make_dig_montage
from mne.fixes import _safe_svd
from mne.io import RawArray
from mne.preprocessing import ICA
from numpy.typing import NDArray
from pymatreader import read_mat


def loader(fname) -> Tuple[ICA, RawArray, NDArray[int]]:
    """Load a file from the MARA dataset and build the MNE ICA instance
    corresponding.
    Parameters
    ----------
    fname : path-like
        Path to the .mat file to read containing the fields:
            - 'cnt' with the ICA activation and the sampling frequency
            - 'mnt' with the montage
            - 'W_ffdiag' with the IC
            - 'goodcomp' with the idx of the brain components
    Returns
    -------
    ica : ICA
        Fitted ICA instance.
    sources : RawArray
        RawArray containing the IC sources as returned by ica.plot_sources().
    brain_components : Array
        Array containing the IDx of the brain components (0-index).
    """
    # load
    mara = read_mat(fname)
    W_ffdiag = mara["W_ffdiag"]
    icaact = mara["cnt"]["x"]
    fs = mara["cnt"]["fs"]
    ch_names = mara["mnt"]["clab"]

    # create montage and info
    ch_pos = dict()
    for k, ch in enumerate(ch_names):
        ch_pos[ch] = mara["mnt"]["pos_3d"][:, k] * 0.095
    montage = make_dig_montage(ch_pos, coord_frame="head")
    info = create_info(ch_names, fs, "eeg")
    info.set_montage(montage)

    # create ICA
    n_components = W_ffdiag.shape[0]
    ica = ICA(method="imported_eeglab", n_components=n_components)
    ica.current_fit = "eeglab"
    ica.ch_names = ch_names
    ica.n_components_ = n_components
    ica.info = info
    ica.pre_whitener_ = np.ones((len(ch_names), 1))
    ica.pca_mean_ = np.zeros(len(ch_names))
    u, s, v = _safe_svd(W_ffdiag)
    ica.unmixing_matrix_ = u * s
    ica.pca_components_ = v
    ica.pca_explained_variance_ = s * s
    ica._update_mixing_matrix()
    ica._update_ica_names()

    # create IC activation / sources
    icaact_names = [f"ICA0{str(k).zfill(2)}" for k in range(n_components)]
    sources = RawArray(icaact.T, create_info(icaact_names, fs, ch_types="misc"))

    # retrieve good components
    brain_components = mara["goodcomp"] - 1
    brain_components = brain_components.astype(int)

    return ica, sources, brain_components
