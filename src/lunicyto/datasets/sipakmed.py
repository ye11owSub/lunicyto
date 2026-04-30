from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

CLASSES: dict[str, int] = {
    "im_Superficial-Intermediate": 0,
    "im_Parabasal": 1,
    "im_Koilocytotic": 2,
    "im_Dyskeratotic": 3,
    "im_Metaplastic": 4,
}

CLASS_NAMES = [
    "Superficial-Intermediate",
    "Parabasal",
    "Koilocytotic",
    "Dyskeratotic",
    "Metaplastic",
]

_IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


def collect_samples(root: str | Path) -> list[tuple[Path, int, str]]:
    """Collect individual cell images from CROPPED/ subdirectories.

    Returns a list of ``(image_path, class_label, slide_group_id)`` triples.
    The ``slide_group_id`` encodes which original microscopy image the cell
    was cropped from (e.g. ``"im_Dyskeratotic_001"`` for ``001_03.bmp``).
    This key is used in :func:`split_samples` to keep all cells from the
    same slide in the same split and thus prevent data leakage.

    Background: each class directory contains two kinds of ``.bmp`` files:
    - **Original images** (e.g. ``001.bmp``) — multi-cell cluster patches.
    - **CROPPED/** — individual cells extracted from those patches
      (e.g. ``001_01.bmp``, ``001_02.bmp``, …).

    Using ``rglob`` on the class directory mixes both types: a training
    sample ``001.bmp`` and a test sample ``001_01.bmp`` would be visually
    nearly identical, inflating test metrics.  This function avoids that
    by reading *only* the ``CROPPED/`` subdirectory.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"dir not found: {root}")

    samples: list[tuple[Path, int, str]] = []

    for class_folder, label in CLASSES.items():
        # Dataset layout: <root>/<class_folder>/<class_folder>/CROPPED/*.bmp
        cropped_dir = root / class_folder / class_folder / "CROPPED"
        if not cropped_dir.exists():
            # Fallback for non-standard layouts: scan whole class dir
            fallback = root / class_folder
            for img_path in sorted(fallback.rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append((img_path, label, img_path.stem))
            continue

        for img_path in sorted(cropped_dir.glob("*")):
            if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                # "001_03.bmp" → slide_id "001" → group "im_Dyskeratotic_001"
                slide_id = img_path.stem.rsplit("_", 1)[0]
                group = f"{class_folder}_{slide_id}"
                samples.append((img_path, label, group))

    return samples


def get_transform(img_size: int, is_train: bool) -> T.Compose:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        return T.Compose(
            [
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                T.RandomGrayscale(p=0.05),
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                normalize,
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ]
    )


class SipakmedDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int, str]],
        transform: T.Compose | None = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _group = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> list[int]:
        return [s[1] for s in self.samples]


def split_samples(
    samples: list[tuple[Path, int, str]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[
    list[tuple[Path, int, str]],
    list[tuple[Path, int, str]],
    list[tuple[Path, int, str]],
]:
    """Stratified slide-level split using StratifiedGroupKFold.

    All cells cropped from the same original microscopy image are kept
    in the same partition (train / val / test).  Class distribution is
    preserved across splits (stratified).  This combination prevents
    both data leakage and AUC computation failures caused by missing
    classes in the validation set.

    n_splits is chosen so that 1 fold ≈ desired test/val fraction:
    - test_split=0.15 → n_splits=7  (1/7 ≈ 14.3%)
    - val_split=0.15  → n_splits=7  (1/7 ≈ 14.3% of train+val)
    """
    labels = np.array([s[1] for s in samples])
    groups = np.array([s[2] for s in samples])
    n = len(samples)

    # Heuristic: pick n_splits so that one fold ≈ requested fraction
    def _n_splits(frac: float) -> int:
        return max(2, round(1.0 / frac))

    # Separate test set — stratified and slide-level
    sgkf_test = StratifiedGroupKFold(
        n_splits=_n_splits(test_split), shuffle=True, random_state=seed
    )
    trainval_idx, test_idx = next(sgkf_test.split(np.zeros(n), labels, groups=groups))

    # Separate val from train+val — also stratified and slide-level
    tv_labels = labels[trainval_idx]
    tv_groups = groups[trainval_idx]
    sgkf_val = StratifiedGroupKFold(
        n_splits=_n_splits(val_split / (1.0 - test_split)), shuffle=True, random_state=seed
    )
    rel_train_idx, rel_val_idx = next(
        sgkf_val.split(np.zeros(len(trainval_idx)), tv_labels, groups=tv_groups)
    )

    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]

    return (
        [samples[i] for i in train_idx],
        [samples[i] for i in val_idx],
        [samples[i] for i in test_idx],
    )


def get_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    all_samples = collect_samples(data_dir)
    train_s, val_s, test_s = split_samples(
        all_samples, val_split=val_split, test_split=test_split, seed=seed
    )

    train_ds = SipakmedDataset(train_s, transform=get_transform(img_size, is_train=True))
    val_ds = SipakmedDataset(val_s, transform=get_transform(img_size, is_train=False))
    test_ds = SipakmedDataset(test_s, transform=get_transform(img_size, is_train=False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def dataset_info(data_dir: str | Path) -> dict:
    samples = collect_samples(data_dir)
    labels = [s[1] for s in samples]
    num_slides = len({s[2] for s in samples})
    per_class = {CLASS_NAMES[i]: labels.count(i) for i in range(len(CLASSES))}

    return {
        "total": len(samples),
        "num_slides": num_slides,
        "per_class": per_class,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASSES),
    }
