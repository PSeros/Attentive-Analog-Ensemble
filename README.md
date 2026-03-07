# Attentive Analog Ensemble (A2E)

Attentive Analog Ensemble (A2E) is a TensorFlow-based framework for probabilistic weather postprocessing. It rethinks the traditional Analog Ensemble (AnEn) by replacing hard analog retrieval with a differentiable cross-attention mechanism, enabling end-to-end learning of analog similarity directly from probabilistic scoring rules such as CRPS and SCRPS.

The framework includes three model variants:

- **A2E** – single-location attentive analog retrieval
- **CA2E** – conditional multi-location variant with shared model weights and location-specific embeddings
- **SA2E** – spatial extension for spatiotemporal modeling across multiple locations

## Motivation

The classical Analog Ensemble is effective for generating probabilistic forecasts, but it has several limitations:

- similarity is based on manually weighted linear predictor combinations
- weight optimization becomes expensive in higher dimensions
- locations are typically modeled independently
- temporal context is short
- analog retrieval is non-differentiable and cannot be trained end-to-end with backpropagation

A2E addresses these limitations by treating:

- the **current forecast** as the **query**
- **historical forecasts** as **keys**
- **historical observations** as **values**

Cross-attention then acts as a soft analog retrieval mechanism, with attention weights defining how strongly each historical observation contributes to the predictive distribution. Because this retrieval is differentiable, the model can learn nonlinear similarity representations in latent space.

## Core idea

A2E combines two main components:

### 1. Embedding Network

A Siamese WaveNet-inspired encoder maps forecast sequences into a shared latent space. The encoder is designed to capture:

- nonlinear feature interactions
- long temporal context
- efficient sequence processing through dilated causal convolutions
- optional conditioning or spatial structure depending on the model variant

### 2. Cross-Attention Retrieval

The query embedding attends to historical forecast embeddings. The resulting weights are used to construct a weighted ensemble of historical observations, which defines the predictive distribution for the target timestep.

## Model variants

### A2E

The base model for single-location probabilistic forecasting. It uses a Siamese encoder and a differentiable cross-attention retrieval layer.

### CA2E

Conditional A2E extends the embedding network with location embeddings, allowing a single model to generalize across multiple locations. This supports transfer learning across sites and makes it easier to add new locations later by learning only their latent representation.

### SA2E

Spatial A2E extends the temporal encoder with spatial convolutions so that the latent state of one location can incorporate information from neighboring or jointly modeled locations.

## Repository structure

The package is organized into modular subpackages:

```
A2E/
    callbacks/
    factory/
    io/
    layer/
    loss/
    metrics/
    model/
    pipeline/
```

These modules cover model definitions, neural layers, training pipelines, callbacks, metrics, loss functions, and high-level workflow orchestration.

## Adding a custom encoder

A2E is designed to be extensible. To add a custom encoder, you need to register it across the model, pipeline, factory, and configuration layers so that it can be instantiated through the standard API.

In general, adding a new encoder requires the following steps:

1. **Implement the encoder layer** in `layer/`  
   Add the encoder implementation as a reusable neural network layer. This layer should define how forecast inputs are mapped into the latent representation used for attention-based retrieval.

2. **Create the model definition** in `model/`  
   Add the model class that uses your new encoder as part of the A2E architecture.

3. **Create the corresponding pipeline** in `pipeline/`  
   Implement the data-preparation pipeline required by your encoder. This includes any model-specific preprocessing, reshaping, or dataset construction.

4. **Register the encoder in the factory** in `factory/factory.py`  
   Extend the factory logic so the framework can instantiate your model and pipeline from the configuration.

5. **Register the configuration** in `io/config.py`  
   Add the required configuration fields and make sure the new encoder or model type can be selected through `ModelConfig`.

After these steps, the encoder should be available through the regular training and inference workflow exposed by the API.

As a rule of thumb:

- use `layer/` for the reusable encoder implementation
- use `model/` for the architecture that integrates the encoder
- use `pipeline/` for data preparation
- use `factory/factory.py` for object creation and dispatch
- use `io/config.py` for user-facing configuration and registration

This design keeps custom extensions aligned with the existing A2E structure and ensures that new encoder types can be used consistently across training, embedding generation, and retrieval workflows.

## Key components

### `model/`

Contains the model implementations for A2E, CA2E, SA2E, and the classical Analog Ensemble baseline. The core A2E model:

1. encodes current and historical forecasts with a shared encoder
2. computes similarity-based attention weights
3. returns aligned historical observations and weights for training against probabilistic losses

### `layer/`

Contains the neural building blocks:

- `cross_attention.py` – differentiable retrieval with configurable similarity metrics
- `encoder.py` – WaveNet-style temporal encoder
- `spatial_encoder.py` – spatiotemporal extension for SA2E

The cross-attention layer supports multiple similarity metrics, including cosine similarity, Pearson correlation, scaled dot product, and Euclidean distance. It also supports optional top-k selection during retrieval.

### `pipeline/`

Builds training datasets and preprocessing workflows for the supported model types. The pipeline layer handles train/test splitting, normalization, and model-specific data preparation.

### `io/`

Provides the main workflow interface through an `Api` class that covers:

- training
- embedding generation
- embedding-based retrieval

It also includes model configuration serialization and plotting utilities.

## Training workflow

The repository exposes a high-level API for training. The general workflow is:

1. create a `ModelConfig`
2. initialize the `Api`
3. prepare forecast and observation tensors
4. provide optimizer, loss, metrics, and training parameters
5. call `train(...)`

The training API supports:

- model creation or loading from disk
- resuming training
- checkpointing and epoch tracking
- saving model config and normalization parameters
- training-history plotting after completion

### Example outline

    import tensorflow as tf
    from A2E.io.config import ModelConfig
    from A2E.io.api import Api

    config = ModelConfig(
        # fill in your model configuration here
    )

    api = Api(config=config)

    model, history = api.train(
        forecasts=forecasts,
        observations=observations,
        optimizer=tf.keras.optimizers.Adam(),
        loss=loss_fn,
        epochs=epochs,
        batch_size=batch_size,
        test_size=0.3,
        save_path="models/a2e_model.keras",
        metrics=metrics,
    )

> Note: adjust imports and configuration fields to match the exact implementation in the repository.

## Evaluation

The framework supports evaluation with:

- **CRPS**
- **SCRPS**
- **RMSE**
- **Bias**
- **Rank histograms** for calibration analysis

These metrics allow assessment of both probabilistic forecast quality and deterministic point accuracy.

## Data

This repository uses [Weather data by Open-Meteo.com](https://open-meteo.com/).

## Why this repository is useful

This repository is intended for researchers and practitioners interested in:

- postprocessing deterministic weather forecasts
- analog ensemble methods
- differentiable retrieval
- similarity learning in latent space

## Citation

If you use this repository in academic work, please cite the associated article:

**Phillip Schlicht**  
_Attentive Analog Ensemble (A2E): End-To-End Learning of Analog Similarity by Optimizing Probabilistic Scoring Rules via Differentiable Retrieval_

> Not published so far.

- Add bibtex when published.
- Use MIT Licence.
