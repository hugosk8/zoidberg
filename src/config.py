from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    PROJECT_NAME: str = "zoidberg"
    SEED: int = 42
    
    IMG_SIZE: tuple[int, int] = (224, 224)
    IMG_CHANNELS: int = 3
    NORMALIZE_PIXELS: bool = True
    CLASS_NAMES: tuple[str, ...] = ("normal", "pneumonia")
    
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.1
    
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    
    @property
    def DATASETS_DIR(self) -> Path:
        return self.BASE_DIR / "datasets"
    
    @property
    def RAW_DIR(self) -> Path:
        return self.DATASETS_DIR / "raw"
    
    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATASETS_DIR / "processed"
    
    @property
    def REPORTS_DIR(self) -> Path:
        return self.BASE_DIR / "reports"
    
    @property
    def FIGURES_DIR(self) -> Path:
        return self.REPORTS_DIR / "figures"
    
    def ensure_dirs(self) -> None:
        self.DATASETS_DIR.mkdir(exist_ok=True)
        self.RAW_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(exist_ok=True)
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
CFG = Config()
CFG.ensure_dirs()