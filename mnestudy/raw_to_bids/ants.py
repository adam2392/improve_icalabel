from pathlib import Path

import mne
from mne_bids import (
    BIDSPath,
    make_dataset_description,
    update_sidecar_json,
    write_raw_bids,
)

mne.set_log_level("WARNING")


# list .fif files TODO: change this when needed
folder = Path("/Users/scheltie/Documents/datasets/iclabel/ANT/raw")
files = [file for file in folder.glob("**/*") if file.suffix == ".fif"]

# BIDS
bids_root = Path("/Users/scheltie/Documents/datasets/iclabel/ANT/bids")
bids_path = BIDSPath(root=bids_root, datatype="eeg", task="neurofeedback")

# load and process files
for k, fname in enumerate(files):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.info["line_freq"] = 50.0

    # drop events and AUX channels
    raw.drop_channels(["TRIGGER", "AUX7", "AUX8"])

    # rename channels with wrong names
    mapping = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "POZ": "POz",
        "FCZ": "FCz",
        "OZ": "Oz",
        "FPz": "Fpz",
    }
    for key, value in mapping.items():
        try:
            mne.rename_channels(raw.info, {key: value})
        except Exception:
            pass

    # add reference channels and montage
    raw.add_reference_channels(ref_channels="CPz")
    raw.set_montage("standard_1020")

    # add device information
    raw.info["device_info"] = dict()
    raw.info["device_info"]["type"] = "EEG"
    raw.info["device_info"]["model"] = "eego mylab"
    raw.info["device_info"]["serial"] = "000479"
    raw.info["device_info"]["site"] = "https://www.ant-neuro.com/products/eego_mylab"

    # add experimenter
    raw.info["experimenter"] = "Mathieu Scheltienne"

    # create a subject id
    subject = str(k)
    # update BIDS path
    bids_path.update(subject=subject)

    # write BIDS
    write_raw_bids(
        raw, bids_path, format="BrainVision", allow_preload=True, overwrite=True
    )

    # update manufacturer field in the sidecar
    bids_path_sidecar = BIDSPath(
        subject=bids_path.subject,
        task=bids_path.task,
        session=bids_path.session,
        suffix="eeg",
        extension=".json",
        root=bids_root,
    )
    update_sidecar_json(bids_path_sidecar, {"Manufacturer": "ANT Neuro"})

# make dataset description
make_dataset_description(
    bids_root,
    name="ANT",
    dataset_type="raw",
    authors="Mathieu Scheltienne",
    overwrite=True,
)