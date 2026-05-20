"""Dataset and file-collection helpers for CNN-based RQA classification."""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from cnn.cnn_utils import parse_metadata_from_filename

class RQAPlotDataset(Dataset):
    """Torch dataset that loads RQA plot images and their labels.

    Parameters
    ----------
    items : list[dict]
        Item records produced by :func:`collect_items`.
    transform : callable or None, default None
        Optional torchvision transform applied to each image.
    """
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of image records.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """Load one image record and return model inputs plus metadata.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            ``(image_tensor, label, subject_id, path)`` for one plot.
        """
        item = self.items[idx]
        img = Image.open(item["filename"]).convert("L")
        if self.transform is not None:
            img = self.transform(img)

        y = int(item["label"])
        subject_id = item["subject_id"]
        path = item["filename"]
        return img, y, subject_id, path
    
def collect_items(root_dir, patient_side=None):
    """Collect PNG plots from a directory tree and attach metadata.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory containing the plot images.
    patient_side : dict or None, default None
        Optional stroke-side lookup passed to
        :func:`cnn.cnn_utils.parse_metadata_from_filename`.

    Returns
    -------
    list[dict]
        One record per image with path, label, condition, and axis metadata.
    """
    root_dir = Path(root_dir)
    items = []

    for p in root_dir.rglob("*.png"):
        meta = parse_metadata_from_filename(p, patient_side=patient_side)
        items.append({
            "path": str(p),
            **meta
        })

    if len(items) == 0:
        raise RuntimeError(f"No .png files found under: {root_dir}")

    return items