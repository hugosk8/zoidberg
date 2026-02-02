from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from src.config import CFG


@dataclass(frozen=True)
class DatasetIndex:
    """Index léger : chemins d'images + labels numériques."""
    paths: list[Path]
    labels: list[int]
    class_to_idx: dict[str, int]
    
    def __len__(self) -> int:
        return len(self.paths)
    

def _iter_image_files(folder: Path) -> Iterable[Path]:
    """Itère sur les fichiers image dans un dossier (récursif)."""
    exts = { ".jpg", ".jpeg", ".png" }
    for p in folder.rglob("*"):
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in exts:
            yield p
    

def _build_class_to_idx(class_names: tuple[str, ...]) -> dict[str, int]:
    return { name: i for i, name in enumerate(class_names) }


def index_split(
    root_dir: Path,
    split: str,
    class_names: tuple[str, ...] = CFG.CLASS_NAMES
) -> DatasetIndex:
    """
    Indexe un split (train/test/val) du dataset:
    root_dir/split/CLASS_NAME/*.jpg

    Retourne:
    - paths: liste de Path vers les images
    - labels: liste d'entiers (0..C-1)
    """
    split_dir = root_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split introuvable: {split_dir}")
    
    class_to_idx = _build_class_to_idx(class_names)
    
    paths: list[Path] = []
    labels: list[int] = []
    
    for class_name in class_names:
        class_dir = split_dir / class_name.upper()
        if not class_dir.exists():
            class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Dossier classe introuvable: {class_dir}")
    
        for img_path in sorted(_iter_image_files(class_dir)):
            paths.append(img_path)
            labels.append(class_to_idx[class_name])
        
    return DatasetIndex(paths=paths, labels=labels, class_to_idx=class_to_idx)


def index_chest_xray(dataset_dir: Path | None = None) -> dict[str, DatasetIndex]:
    if dataset_dir is None:
        dataset_dir = CFG.RAW_DIR / "chest_xray"
        
    splits = {}
    for split in ("train", "val", "test"):
        splits[split] = index_split(dataset_dir, split)
        
    return splits