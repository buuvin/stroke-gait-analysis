"""Shared utilities for CNN-based RQA image classification."""

from pathlib import Path
import numpy as np
from PIL import Image
import random 
import torch


def set_seed(seed=42):
    """Seed the Python, NumPy, and Torch RNGs.

    Parameters
    ----------
    seed : int, default 42
        Seed value used for deterministic behavior.

    Returns
    -------
    None
        The RNG states are updated in place.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

patient_side = {
    "CK01":	"left",
    "CK02":	"left",
    "CK03":	"right",
    "CK05":	"left",
    "CK06":	"right",
    "CK07":	"right",
    "CK08":	"right",
    "CK09":	"left",
    "CK10":	"right",
    "CK11":	"left",
    "CK12":	"right",
    "CK13":	"right",
    "CK14":	"left",
    "CK15":	"left",
}

def parse_metadata_from_filename(file_path, patient_side):
    """Parse subject/condition metadata from a COP filename.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Name or path of one COP text file.
    patient_side : dict
        Mapping from stroke subject ID to affected limb side.

    Returns
    -------
    dict
        Metadata fields used for feature records and merge keys.

    Notes
    -----
    Healthy subjects are encoded with an empty ``affected_side`` string.
    """
    name = Path(file_path).name

    tokens = name.split("_")
    subject_id = tokens[0]

    if subject_id.startswith("CK"):
        label = 1
        condition = "stroke"
        affected_side = patient_side[subject_id] + "_affected"
    elif subject_id.startswith("SUP"):
        label = 0
        condition = "healthy"
        affected_side = ""
    else:
        raise ValueError(f"Unknown subject type: {subject_id}")

    if "PSEO" in name:
        eyes = "eyes_open"
    elif "PSEC" in name:
        eyes = "eyes_closed"
    else:
        eyes = "unknown"

    if "COP1" in name:
        cop_plate = "left"
    elif "COP2" in name:
        cop_plate = "right"
    else:
        cop_plate = "combined"

    if name.endswith("_X.txt"):
        axis = "x"
    elif name.endswith("_Y.txt"):
        axis = "y"
    else:
        axis = "resultant"

    return {
        "filename": name,
        "subject_id": subject_id,
        "label": label,
        "category": condition,
        "eye_condition": eyes,
        "cop_type": cop_plate,
        "affected_side": affected_side,
        "axis": axis
    }

def is_empty_plot(path, resize=256, white_thresh=245, min_ink_frac=0.005, min_var=15.0):
    """Detect plots that are likely empty or near-empty.

    Parameters
    ----------
    path : str or pathlib.Path
        Image path.
    resize : int, default 256
        Side length used for the grayscale check.
    white_thresh : int, default 245
        Pixel intensity threshold used to count foreground ink.
    min_ink_frac : float, default 0.005
        Minimum non-white pixel fraction required to keep the image.
    min_var : float, default 15.0
        Minimum grayscale variance required to keep the image.

    Returns
    -------
    bool
        ``True`` when the image should be treated as empty.
    """
    img = Image.open(path).convert("L").resize((resize, resize))
    arr = np.array(img, dtype=np.uint8)

    var = float(arr.var())
    ink_frac = float((arr < white_thresh).mean())

    return (ink_frac < min_ink_frac) or (var < min_var)