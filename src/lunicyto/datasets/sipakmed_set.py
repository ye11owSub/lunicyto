from pathlib import Path
from typing import Iterable
from pydantic import BaseModel


class DatasetInfo(BaseModel):
    total: int
    class_items_count: dict[str, int]

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


def collect_samples(src: str | Path) -> list[tuple[Path, int]]:
    """
    Recursively collects all images in `src`, assigning labels based on the name
    of the top-level parent folder.

    Returns a list of pairs (image_path, class_label).
    """
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Directory not found: {src}")

    samples: list[tuple[Path, int]] = []

    for class_folder, label in CLASSES.items():
        class_dir = src / class_folder
        if not class_dir.exists():
            continue
        # rglob recursively traverses nested folders (im_X/im_X/*.bmp, CROPPED/*.bmp)
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                samples.append((img_path, label))

    if not samples:
        raise FileNotFoundError(
            f"Images not found in {src}. Expected subfolders: {list(CLASSES.keys())}"
        )

    return samples


def dataset_info(data_dir: str | Path) -> DatasetInfo:
    samples = collect_samples(data_dir)
    labels = [s[1] for s in samples]
    class_items_count = {CLASS_NAMES[i]: labels.count(i) for i in range(len(CLASSES))}

    return DatasetInfo(total=len(samples), class_items_count=class_items_count)
