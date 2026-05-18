from __future__ import annotations

from pathlib import Path
from typing import TypedDict

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
WIND_DIR = DATA_DIR / "wind"
GERMANY_SHAPE_DIR = WIND_DIR / "germany_shape"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
FIGURES_DIR = EVALUATION_DIR / "figures"
ABLATION_STUDY_DIR = EVALUATION_DIR / "AblationStudy"

TRAINED_MODELS_DIR = PROJECT_ROOT / "Trained_Models"

COORDINATES_CSV = WIND_DIR / "coordinates.csv"
COORDINATES_PNG = WIND_DIR / "coordinates.png"
GERMANY_SHP = GERMANY_SHAPE_DIR / "de.shp"

METRICS_EVALUATION_CSV = EVALUATION_DIR / "metrics_evaluation.csv"
ABLATION_METRICS_EVALUATION_CSV = ABLATION_STUDY_DIR / "metrics_evaluation.csv"

OBSERVATIONS_WIND_SPEED_CSV = WIND_DIR / "observations_wind_speed.csv"
OBSERVATIONS_WIND_DIRECTION_CSV = WIND_DIR / "observations_wind_direction.csv"


class ModelPathBundle(TypedDict):
    directory: Path
    model: Path
    config: Path
    evaluation: Path


def wind_file(filename: str) -> Path:
    """Return a file path inside data/wind."""
    return WIND_DIR / filename


def evaluation_file(*parts: str) -> Path:
    """Return a file path inside evaluation."""
    return EVALUATION_DIR.joinpath(*parts)


def model_dir(model_type: str, location: int | None = None) -> Path:
    """Return the directory for a model type, optionally scoped to a location."""
    base = TRAINED_MODELS_DIR / model_type
    return base / f"Location{location}" if location is not None else base


def model_paths(model_type: str, model_name: str, location: int | None = None) -> ModelPathBundle:
    """Return common paths for a trained model and its config/evaluation folder."""
    directory = model_dir(model_type, location)
    return {
        "directory": directory,
        "model": directory / f"{model_name}.keras",
        "config": directory / f"{model_name}_config.json",
        "evaluation": directory / f"{model_name}_evaluation",
    }


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for `path` and return `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
