#%% Imports

import numpy as np
from mne import create_info
from mne.channels import make_dig_montage
from mne.fixes import _safe_svd
from mne.io import RawArray
from mne.preprocessing import ICA
from pymatreader import read_mat


#%% Read .mat file
fname = r""
mara = read_mat(fname)
icaact = mara["cnt"]["x"]
fs = mara["cnt"]["fs"]
ch_names = mara["mnt"]["clab"]
W_ffdiag = mara["W_ffdiag"]

#%% Montage
#
# sphere radius in MNE is 0.095m by default, let's match the *7 and *8
# electrodes on this radius with the default (x, y, z, radius) sphere to get
# an EEGLAB-like (MATLAB) projection.
#
# .. code-block:: python
#
#       ch_on_sphere = [
#           "AF7", "F7", "FT7", "T7", "TP7", "P7", "PO7",
#           "AF8", "F8", "FT8", "T8", "TP8", "P8", "PO8",
#       ]
#
# by default, np.linalg.norm(mara['mnt']['pos_3d'][:, k]) for channels in
# ch_on_sphere is 1!

ch_pos = dict()
for k, ch in enumerate(ch_names):
    ch_pos[ch] = mara["mnt"]["pos_3d"][:, k] * 0.095
montage = make_dig_montage(ch_pos, coord_frame="head")

#%% Create Info

info = create_info(ch_names, fs, "eeg")
info.set_montage(montage)

#%% Create ICA
#
# Uses as similar approach to the loading of ICA instance from EEGLAB, but
# applied to ``W_ffdiag``.
# c.f. https://github.com/mne-tools/mne-python/blob/main/mne/preprocessing/ica.py#L2794-L2856

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
# Really not sure about the PCA ones.. the MARA dataset was obtained after
# applying a PCA to reduce from 115-ish to 30 components.
ica.pca_components_ = v
ica.pca_explained_variance_ = s * s
ica._update_mixing_matrix()
ica._update_ica_names()

#%% Create sources - ICA activations
icaact_names = [f"ICA0{str(k).zfill(2)}" for k in range(n_components)]
sources = RawArray(icaact.T, create_info(icaact_names, fs, ch_types="misc"))

#%% Retrieve good components
#
# The good components range from 1 to 30, i.e. they are 1-index, so let's
# substract 1 to retrieve the 0-index indices.

brain_components = mara["goodcomp"] - 1