from mne.datasets import sample
from mne.io import read_raw
from mne.preprocessing import ICA
import  numpy as np

directory = sample.data_path() / "MEG" / "sample"
raw = read_raw(directory / "sample_audvis_raw.fif", preload=False)
raw.pick_types(eeg=True)
# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True)
raw.load_data()

# ica = ICA(n_components=5, method="picard")  # the script will not work if we choose n_componeents less than number of channels
ica = ICA(method="picard")
ica.fit(raw)

data = raw.get_data()

###################################
ica_times_Series = ica.get_sources(raw)._data
mixing_matrix = ica.mixing_matrix_
n = ica.pca_components_[0]
one_ic_timeseries = ica_times_Series[0]
raw = raw.get_data()
#######################################

# Covariance matrix of the raw data
R = np.cov(raw)  # equation 7
u, s, vh = np.linalg.svd(R, full_matrices=True)  # equation - 8

# raw_data_signal_subspace = np.dot(((1/np.sqrt(s)) * vh), raw)  -- ignore

# since we already have IC time series  so we can skip the steps to find 's(tk)'
# we can plug in all the values in equation 17

# equation 17
x  = np.dot(np.dot(np.dot(u,np.sqrt(s)),mixing_matrix),ica_times_Series) # How to get time series for all channels i.e.59
