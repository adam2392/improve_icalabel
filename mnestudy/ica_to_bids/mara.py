import os
from collections import OrderedDict
from pathlib import Path

from mara_loader import loader
from mne_bids import BIDSPath
from mne_bids.config import BIDS_COORD_FRAME_DESCRIPTIONS, MNE_TO_BIDS_FRAMES
from mne_bids.utils import _write_json, _write_tsv
from mne_icalabel.annotation import mark_component, write_components_tsv


def convert(directory_in, directory_out):
    """Convert the MARA dataset to BIDS format.

    Parameters
    ----------
    directory_in : path-like
        Path to the directory containing the MARA dataset.
    directory_out : path-like
        Path to the directory where the BIDS dataset is saved.
    """
    directory_in, directory_out = _check_paths(directory_in, directory_out)
    bids_path = BIDSPath(root=directory_out, datatype="eeg")

    # keep a dict mapping of identifiers and 0 to N idx
    mapping = dict()  # identifier: idx
    inc = 1

    for fname in directory_in.iterdir():
        if fname.is_dir() or fname.suffix != ".mat":
            continue
        print(f"\n\nConverting {fname}...")

        identifier = fname.stem.split("oddball_fasor_")[1].replace("_", "")
        ica, sources, brain_components = loader(fname)
        print("Finished loading data... \n\n")
        # update BIDSPath
        if identifier not in mapping:
            mapping[identifier] = str(inc)
            inc += 1
        bids_path.update(subject=mapping[identifier], task=identifier, extension=".fif")

        # write files
        bids_path.update(processing="ica")
        ica.save(bids_path.fpath.with_suffix(".fif"), overwrite=True)
        bids_path.update(processing="sources")
        sources.save(bids_path.fpath.with_suffix(".fif"), overwrite=True)

        # write good and bad components
        comp_bids_path = bids_path.copy().update(
            processing="iclabels", suffix="ica", extension=".tsv", check=False
        )
        print("\n\n Trying to write componnets tsv to: ", comp_bids_path)
        write_components_tsv(ica, comp_bids_path)
        for k in range(ica.n_components_):
            label = "brain" if k in brain_components else "noise"
            mark_component(
                k,
                comp_bids_path,
                method="MARA",
                label=label,
                author="MARA",
                strict_label=False,
            )

        # write montage
        _write_montage(bids_path, ica.info.get_montage())


def _check_paths(directory_in, directory_out):
    """Check that the path exists."""
    directory_in = Path(directory_in)
    directory_out = Path(directory_out)
    if not directory_in.exists():
        raise ValueError("The provided directory 'direcotry_in' does not exist.")
    if not directory_out.exists():
        os.makedirs(directory_out)
    return directory_in, directory_out


def _write_montage(bids_path, montage):
    """Write the montage to the bids-path.

    Fiducials nasion, lpa and rpa are missing.
    """
    # sanity-check and retrieve coordinate frame
    pos = montage.get_positions()
    assert all(pos[fid_key] is None for fid_key in ("nasion", "lpa", "rpa"))
    assert int(montage.dig[0]["coord_frame"]) == 4  # 'head'
    mne_coord_frame = "head"
    coord_frame = MNE_TO_BIDS_FRAMES.get(mne_coord_frame, None)

    coord_file_entities = {
        "root": bids_path.root,
        "datatype": bids_path.datatype,
        "subject": bids_path.subject,
        "session": bids_path.session,
        "acquisition": bids_path.acquisition,
        "space": coord_frame,
    }
    channels_path = BIDSPath(
        **coord_file_entities,
        suffix="electrodes",
        extension=".tsv",
    )
    coordsystem_path = BIDSPath(
        **coord_file_entities, suffix="coordsystem", extension=".json"
    )

    # write channel list and coordinates
    x, y, z, names = list(), list(), list(), list()
    for ch, coordinates in montage.get_positions()["ch_pos"].items():
        assert not any(elt is None for elt in coordinates)
        names.append(ch)
        x.append(coordinates[0])
        y.append(coordinates[1])
        z.append(coordinates[2])
    data = OrderedDict(
        [
            ("name", names),
            ("x", x),
            ("y", y),
            ("z", z),
        ]
    )
    _write_tsv(channels_path, data, overwrite=True)

    # write coordinate system
    coords = dict(
        NAS=list(),
        LPA=list(),
        RPA=list(),
    )
    sensor_coord_system_descr = BIDS_COORD_FRAME_DESCRIPTIONS.get(
        mne_coord_frame, "n/a"
    )
    fid_json = {
        "EEGCoordinateSystem": mne_coord_frame,
        "EEGCoordinateUnits": "m",  # default
        "EEGCoordinateSystemDescription": sensor_coord_system_descr,
        "AnatomicalLandmarkCoordinates": coords,
        "AnatomicalLandmarkCoordinateSystem": mne_coord_frame,
        "AnatomicalLandmarkCoordinateUnits": "m",  # default
    }
    _write_json(coordsystem_path, fid_json, overwrite=True)


if __name__ == "__main__":
    from mara import convert

    directory_in = Path("/Users/adam2392/Downloads/fasor/")
    directory_out = Path("/Users/adam2392/Downloads/fasor_bids/")

    convert(directory_in, directory_out)
