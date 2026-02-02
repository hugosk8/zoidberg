from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collections import Counter
from src.data.load import index_chest_xray


def main() -> None:
    splits = index_chest_xray()

    for split_name, ds in splits.items():
        print(f"\n=== {split_name} ===")
        print("n_images:", len(ds))
        print("classes:", ds.class_to_idx)

        # Vérif paths
        assert len(ds.paths) == len(ds.labels), "paths et labels doivent avoir la même taille"
        assert len(ds) > 0, f"{split_name} est vide"

        # Vérif fichiers existants
        missing = [p for p in ds.paths if not p.exists()]
        assert not missing, f"Fichiers manquants (ex: {missing[:3]})"

        # Vérif labels
        allowed = set(ds.class_to_idx.values())
        assert set(ds.labels).issubset(allowed), "labels hors mapping"

        # Distribution
        counts = Counter(ds.labels)
        inv = {v: k for k, v in ds.class_to_idx.items()}
        pretty = {inv[k]: v for k, v in counts.items()}
        print("distribution:", pretty)

    print("\nOK ✅ Loader fonctionnel.")


if __name__ == "__main__":
    main()
