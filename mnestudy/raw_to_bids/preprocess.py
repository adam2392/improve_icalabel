import itertools

import mne
import pyprep
from mne.preprocessing import ICA, compute_bridged_electrodes


def preprocess_ANT_dataset(raw):
    """Automatic preprocessing pipeline for ANT dataset."""
    raw = raw.copy()
    # Bandpass standard filter
    # 100 Hz edge should retain muscle activity.
    bandpass = (1.0, 100.0)  # Hz
    raw.filter(
        l_freq=bandpass[0],
        h_freq=bandpass[1],
        picks="eeg",
        method="fir",
        phase="zero-double",
        fir_window="hamming",
        fir_design="firwin",
        pad="edge",
    )
    # Set montage
    raw.set_montage("standard_1020")

    # Detect bad channels, on a copy
    nc = pyprep.find_noisy_channels.NoisyChannels(raw.copy())
    nc.find_all_bads()
    raw.info["bads"] = nc.get_bads()

    # Look for bridged electrodes
    # requires MNE >= 1.1.0
    bridged_idx, _ = compute_bridged_electrodes(raw)
    bridged_ch_names = [raw.ch_names[k] for k in itertools.chain(*bridged_idx)]
    raw.info["bads"] = list(set(raw.info["bads"] + bridged_ch_names))

    # CAR reference
    # The reference channel CPz is not added as it would just introduce an
    # additional dependency and is not required for the ICA.
    # The CAR reference excludes the bad channels.
    raw.set_eeg_reference("average", ch_type="eeg", projection=False)

    # Fit ICA decomposition on good channels
    # Fit on n_good_channels-1 because of CAR reference.
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    ica = ICA(n_components=picks.size - 1, method="picard")
    ica.fit(raw, picks=picks)

    return raw, ica
