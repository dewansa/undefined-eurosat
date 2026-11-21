"""Teacher only: build the train and test files from the original EuroSAT download.

    uv run python tools/make_data.py path/to/EuroSAT_RGB

EuroSAT RGB (MIT license, https://zenodo.org/records/7711810) has 27,000 satellite
photos, 64x64 pixels, in one folder per class. This script:

1. Splits each class 80/20 with a fixed seed, so both sets have the same class mix.
2. Writes the training images and labels to dist/eurosat-train.npz.
3. Writes the test images, shuffled and WITHOUT labels, to dist/eurosat-test.npz.
4. Writes the test labels to private/test-labels.txt (git-ignored). They go into the
   repo secret TEST_LABELS, and only the grading job reads them.

Upload both .npz files to the GitHub release "data-v1"; src/eurosat/data.py downloads
them from there and checks their SHA-256, printed at the end.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 2026
TEST_SHARE = 0.2
CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def load_class(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.jpg"))
    images = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    assert images.shape[1:] == (64, 64, 3), images.shape
    return images.transpose(0, 3, 1, 2)  # (n, 3, 64, 64): channels first, as PyTorch expects


def main() -> None:
    source = Path(sys.argv[1])
    rng = np.random.default_rng(SEED)
    train_x, train_y, test_x, test_y = [], [], [], []
    for label, name in enumerate(CLASSES):
        images = load_class(source / name)
        order = rng.permutation(len(images))
        n_test = round(len(images) * TEST_SHARE)
        test_x.append(images[order[:n_test]])
        train_x.append(images[order[n_test:]])
        test_y += [label] * n_test
        train_y += [label] * (len(images) - n_test)
        print(f"{name:22} {len(images) - n_test:5} train {n_test:5} test")

    # Shuffle so neither file is sorted by class.
    train_x, train_y = np.concatenate(train_x), np.array(train_y, dtype=np.uint8)
    test_x, test_y = np.concatenate(test_x), np.array(test_y, dtype=np.uint8)
    order = rng.permutation(len(train_x))
    train_x, train_y = train_x[order], train_y[order]
    order = rng.permutation(len(test_x))
    test_x, test_y = test_x[order], test_y[order]

    # The original has a few exact duplicates. One copy in each set would leak test answers.
    train_hashes = {hashlib.sha1(x.tobytes()).digest() for x in train_x}
    leaked = sum(hashlib.sha1(x.tobytes()).digest() in train_hashes for x in test_x)
    print(f"Test images that also appear in train: {leaked}")

    dist, private = Path("dist"), Path("private")
    dist.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    np.savez_compressed(dist / "eurosat-train.npz", images=train_x, labels=train_y)
    np.savez_compressed(dist / "eurosat-test.npz", images=test_x)
    (private / "test-labels.txt").write_text("".join(map(str, test_y)) + "\n")
    for path in sorted(dist.glob("*.npz")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path}: {path.stat().st_size / 1e6:.0f} MB, sha256 {digest}")


if __name__ == "__main__":
    main()
