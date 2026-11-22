"""Load the EuroSAT satellite images.

Every image is a 3x64x64 color photo (red, green, blue) of a patch of land in Europe,
taken by the Sentinel-2 satellite. Every label is a number from 0 to 9 (see CLASSES).

The images are kept as one uint8 tensor (values 0 to 255), which takes 4 times less
memory than floats. Indexing the dataset gives a float image with values from 0 to 1,
like Fashion-MNIST did. For speed, use `images` and `labels` directly.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def find_data_dir(start: Path | None = None) -> Path:
    """The repo's data/ folder, found from any folder inside the repo.

    Notebooks run inside submissions/<you>/, so a plain "data" would download the
    images a second time there.
    """
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").is_file() and (folder / "submissions").is_dir():
            return folder / "data"
    return start / "data"


DATA_DIR = find_data_dir()
RELEASE_URL = "https://github.com/dewansa/undefined-eurosat/releases/download/data-v1"
FILES = {
    "eurosat-train.npz": "fe3d3781b6ee4750c8432e61607ffc9201aba092c5a9cbcd8ecc2f821d955c7a",
    "eurosat-test.npz": "bbd6c7e346919e02b03e39f49869654798f35f41d716edc6bcdf7167d48b3cfe",
}

CLASSES = (
    "Annual crop",
    "Forest",
    "Herbaceous vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Permanent crop",
    "Residential",
    "River",
    "Sea or lake",
)

# For `uv run score` on your computer: these training images are held back from
# train() and used to check the model, because the real test labels are secret.
CHECK_SHARE = 0.2
CHECK_SEED = 7


class EuroSAT(Dataset):
    """Images, and labels when they are known (the test labels are not)."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor | None) -> None:
        self.images = images  # uint8, shape (n, 3, 64, 64)
        self.labels = labels  # int64, shape (n,), or None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index].float() / 255
        label = -1 if self.labels is None else int(self.labels[index])
        return image, label

    def subset(self, indices: torch.Tensor) -> EuroSAT:
        labels = None if self.labels is None else self.labels[indices]
        return EuroSAT(self.images[indices], labels)


def download(name: str, root: Path = DATA_DIR) -> Path:
    """Download one data file once, and check it is complete."""
    path = root / name
    if path.is_file() and sha256(path) == FILES[name]:
        return path
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} (about 240 MB for both files, only the first time)...")
    partial = path.with_suffix(".part")
    urllib.request.urlretrieve(f"{RELEASE_URL}/{name}", partial)
    if sha256(partial) != FILES[name]:
        partial.unlink()
        raise RuntimeError(f"{name} did not download correctly. Please try again.")
    partial.replace(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def train_set(root: Path = DATA_DIR) -> EuroSAT:
    """The 21,600 training images and labels. Students may use these however they like."""
    with np.load(download("eurosat-train.npz", root)) as file:
        images = torch.from_numpy(file["images"])
        labels = torch.from_numpy(file["labels"]).long()
    return EuroSAT(images, labels)


def test_set(root: Path = DATA_DIR) -> EuroSAT:
    """The 5,400 test images, without labels. Only the scoring script uses these."""
    with np.load(download("eurosat-test.npz", root)) as file:
        return EuroSAT(torch.from_numpy(file["images"]), None)


def split_for_check(
    full: EuroSAT, seed: int = CHECK_SEED, share: float = CHECK_SHARE
) -> tuple[EuroSAT, EuroSAT]:
    """Split the training set into (images to train on, images to check with).

    The same seed gives the same split every time, so local scores are comparable
    between runs. `uv run score` uses the default seed. Choose your settings on a
    split with another seed, then confirm them with `uv run score`: a setting picked
    because it scored well on some images is lucky on exactly those images.
    """
    order = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))
    n_check = round(len(full) * share)
    return full.subset(order[n_check:]), full.subset(order[:n_check])
