import inspect
import tensorflow as tf
from typing import Optional, List, Any, Dict, Callable
from A2E.io.config import ModelConfig
from A2E.model.mini_a2e import MiniA2E
from A2E.pipeline.a2e import A2EPipeline
from A2E.pipeline.ca2e import CA2EPipeline
from A2E.pipeline.sa2e import SA2EPipeline
from A2E.model.a2e import A2E
from A2E.model.ca2e import CA2E
from A2E.model.sa2e import SA2E

PIPELINE_DISTRIBUTION = {
    "MiniA2E": A2EPipeline,
    "A2E": A2EPipeline,
    "CA2E": CA2EPipeline,
    "SA2E": SA2EPipeline,
}

MODEL_DISTRIBUTION = {
    "MiniA2E": MiniA2E,
    "A2E": A2E,
    "CA2E": CA2E,
    "SA2E": SA2E,
}

INPUT_SHAPE_CONFIGS: Dict[str, Callable[[ModelConfig], List[tuple]]] = {
    "MiniA2E": lambda cfg: [
        (None, cfg.seq_len, cfg.d_vars),  # current forecasts
        (None, cfg.lookback, cfg.d_vars),  # historical forecasts
        (None, cfg.lookback, 1)  # historical observations
    ]
    ,
    "A2E": lambda cfg: [
            (None, cfg.seq_len, cfg.d_vars),          # current forecasts
            (None, cfg.lookback, cfg.d_vars),         # historical forecasts
            (None, cfg.lookback, 1)                   # historical observations
        ]
    ,
    "CA2E": lambda cfg: [
        (None, cfg.seq_len, cfg.d_vars),              # current forecasts
        (None, cfg.lookback, cfg.d_vars),             # historical forecasts
        (None, cfg.lookback, 1),                      # historical observations
        (None,)                                       # location
    ],
    "SA2E": lambda cfg: [
        (None, cfg.seq_len, cfg.d_loc, cfg.d_vars),   # current forecasts
        (None, cfg.lookback, cfg.d_loc, cfg.d_vars),  # historical forecasts
        (None, cfg.lookback, 1)                       # historical observations
    ]
}

class Factory:
    """Factory class for creating A2E models. And their data Pipelines"""
    def __init__(self, config: ModelConfig):
        self.config = config

    def build_model(
            self,
            optimizer: Optional[tf.keras.optimizers.Optimizer],
            loss: Optional[tf.keras.losses.Loss],
            metrics: Optional[List[tf.keras.metrics.Metric]],
            **kwargs) -> tf.keras.Model:
        """
        Fetches and creates any A2E model.
        """
        # Fetch model, params and shapes
        model_class = self._get_model_class()
        model_params = self._get_model_params(model_class, **kwargs)
        input_shapes = self._get_input_shape()
        # Initialize, build and compile
        model = model_class(**model_params)
        model.build(input_shape=input_shapes)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])
        return model

    def build_pipeline(self, forecasts,observations, test_size, batch_size):
        pipeline_class = PIPELINE_DISTRIBUTION[self.config.model_type]
        pipeline = pipeline_class(
            forecasts=forecasts,
            observations=observations,
            config=self.config,
            test_size=test_size,
            batch_size=batch_size
        )
        return pipeline

    def _get_model_class(self):
        return MODEL_DISTRIBUTION[self.config.model_type]

    def _get_model_params(self, model_class, **kwargs) -> Dict[str, Any]:
        """
        Automatically extract parameters from config, needed for a specific model.

        Returns:
            Dictionary with filtered parameters
        """
        # Analyze signature of model's __init__ method
        sig = inspect.signature(model_class.__init__)
        model_params = set(sig.parameters.keys())

        # Get all config parameters
        config_dict = {k: v for k, v in self.config.__dict__.items() if v}

        # Include parameters from kwargs
        kwargs_dict = {k: v for k, v in kwargs.items() if k in model_params}
        combined_dict = {**config_dict, **kwargs_dict}

        # Return only parameters that the model actually needs
        filtered_params = {k: v for k, v in combined_dict.items() if k in model_params}
        return filtered_params

    def _get_input_shape(self):
        return INPUT_SHAPE_CONFIGS[self.config.model_type](self.config)
