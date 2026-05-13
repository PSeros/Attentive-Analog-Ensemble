import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Literal

ModelType = Literal[
    "MiniA2E",
    "A2E",
    "CA2E",
    "SA2E",
    "SCA2E"
]
SimilarityType = Literal[
    "cosine_similarity",
    "pearson_correlation",
    "scaled_dot_product",
    "euclidean_distance"
]

@dataclass
class ModelConfig:
    """Configuration for A2E models."""
    model_type: ModelType
    similarity_metric: SimilarityType
    d_model: int
    n_blocks: int
    seq_len: int
    lookback: int
    time_to_target: int
    d_vars: int
    foresight: int
    dropout: float = 0

    # CA2E specific parameters
    max_locations: Optional[int] = None

    # SA2E specific parameters
    d_loc: Optional[int] = None
    locations_kernel_size: Optional[int] = None

    # Training parameters (Used to track the total number of epochs)
    epochs: int = None

    # Normalization parameters (set during training)
    forecast_normalizer_mean: Optional[List[float]] = None
    forecast_normalizer_variance: Optional[List[float]] = None
    observation_normalizer_mean: Optional[List[float]] = None
    observation_normalizer_variance: Optional[List[float]] = None

    def to_dict(self):
        """Convert ModelConfig to a dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """Create a ModelConfig instance from a dictionary, ignoring extra keys."""
        filtered_dict = {k: v for k, v in config_dict.items() if k in cls.__annotations__}
        return cls(**filtered_dict)

    def save_to_json(self, file_path: str) -> None:
        """
        Save the ModelConfig to a JSON file.

        Args:
            file_path: Path where the JSON file will be saved
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert to dictionary and save as JSON
        config_dict = self.to_dict()

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str) -> 'ModelConfig':
        """
        Load a ModelConfig from a JSON file.

        Args:
            file_path: Path to the JSON file containing the configuration

        Returns:
            ModelConfig instance loaded from the JSON file

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
            TypeError: If the JSON contains invalid configuration parameters
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)
