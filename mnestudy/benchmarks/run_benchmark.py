import json

from mne.preprocessing import read_ica
from mne_bids import BIDSPath, read_raw_bids
from mne_icalabel.gui import label_ica_components
from mne_icalabel.iclabel import get_iclabel_features
from sklearn.ensemble import RandomForestClassifier


def load_data(bids_path, raw_path):
    pass


def main():
    # where the data for the ICA is
    data_dir = ""
    bids_root = ""
    subject = ""
    session = ""
    extension = ""  # the extension for the raw data

    # set up a path object to where the data is
    bids_path = BIDSPath(
        root=data_dir,
        subject=subject,
        session=session,
        suffix="ica",
        extension=".fif",
        check=False,
    )

    # Optionally: get the corresponding raw file if we have access to the raw data
    raw_path = bids_path.copy().update(
        root=bids_root, suffix="eeg", extension=extension
    )

    # read the ICA
    ica = read_ica(bids_path)

    # read the raw file
    raw = read_raw_bids(raw_path)

    # now run the labeling components ICA
    # gui = label_ica_components(ica, raw)


def preprocess_ica(ica, raw):
    # extract features of the ICA instance
    ica_features = get_iclabel_features(raw, ica)


if __name__ == "__main__":
    main()
