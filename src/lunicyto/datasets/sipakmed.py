from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


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

def collect_samples(root: str | Path) -> List[Tuple[Path, int]]:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"dir not found: {root}")

    samples: List[Tuple[Path, int]] = []

    for class_folder, label in CLASSES.items():
        class_dir = root / class_folder
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append((img_path, label))

    return samples


def get_transform(img_size: int, is_train: bool) -> T.Compose:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        return T.Compose([
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
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        normalize,
    ])


class SipakmedDataset(Dataset):

    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        transform: Optional[T.Compose] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def labels(self) -> List[int]:
        return [s[1] for s in self.samples]


def split_samples(
    samples: List[Tuple[Path, int]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
    List[Tuple[Path, int]],
]:

    labels = np.array([s[1] for s in samples])
    n = len(samples)

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(n), labels))

    val_frac_of_trainval = val_split / (1.0 - test_split)
    labels_tv = labels[train_val_idx]
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_of_trainval, random_state=seed
    )
    rel_train_idx, rel_val_idx = next(sss_val.split(np.zeros(len(train_val_idx)), labels_tv))

    train_idx = train_val_idx[rel_train_idx]
    val_idx = train_val_idx[rel_val_idx]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    return train_samples, val_samples, test_samples


def get_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

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
    per_class = {CLASS_NAMES[i]: labels.count(i) for i in range(len(CLASSES))}

    return {
        "total": len(samples),
        "per_class": per_class,
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASSES),
    }
