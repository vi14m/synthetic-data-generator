import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sdv.single_table import CopulaGANSynthesizer as CopulaGAN
from sdv.metadata import SingleTableMetadata
from .base_generator import BaseGenerator


class CopulaGANGenerator(BaseGenerator):
    """CopulaGAN generator for synthetic tabular data."""

    def __init__(self, metadata: Optional[SingleTableMetadata] = None, **kwargs):
        """
        Initialize the CopulaGAN generator.

        Args:
            metadata (Optional[SingleTableMetadata]): Metadata object for the data.
            **kwargs: Additional arguments to pass to the CopulaGAN constructor.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_kwargs = kwargs
        self.metadata = metadata
        self.epochs = 300
        self.batch_size = 500
        self.embedding_dim = 128

    def configure(self, epochs: int = 300, batch_size: int = 500, embedding_dim: int = 128, **kwargs):
        """
        Configure the CopulaGAN generator.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            embedding_dim: Dimension of the embedding layer.
            **kwargs: Additional configuration parameters.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured CopulaGAN with epochs={epochs}, batch_size={batch_size}, embedding_dim={embedding_dim}")

    def fit(self, data: pd.DataFrame):
        """
        Fit the CopulaGAN model to the input data.

        Args:
            data: Input DataFrame to learn from.

        Returns:
            self: The fitted generator.
        """
        try:
            if not self.validate_data(data):
                raise ValueError("Input data is invalid or empty.")

            # Use existing metadata or infer from data
            self.metadata = self.metadata or self._get_metadata(data)

            self.model = CopulaGAN(
                metadata=self.metadata,
                enforce_min_max_values=True,
                enforce_rounding=False,
                numerical_distributions=self.model_kwargs.get('numerical_distributions', None),
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                verbose=True,
                **self.model_kwargs
            )

            self.logger.info(f"Fitting CopulaGAN model on data with shape {data.shape}")
            self.model.fit(data)
            self.logger.info("CopulaGAN model fitting completed")

            return self

        except ImportError:
            self.logger.error("SDV package not found. Please install it with 'pip install sdv'")
            raise
        except Exception as e:
            self.logger.error(f"Error fitting CopulaGAN model: {str(e)}")
            raise

    def generate(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic samples using the fitted CopulaGAN model.

        Args:
            num_samples: Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: DataFrame containing the generated synthetic samples.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        try:
            self.logger.info(f"Generating {num_samples} synthetic samples")
            synthetic_data = self.model.sample(num_samples)
            self.logger.info(f"Generated synthetic data with shape {synthetic_data.shape}")
            return synthetic_data

        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def save(self, path: str):
        """
        Save the fitted CopulaGAN model to disk.

        Args:
            path: Path to save the model to.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            self.logger.info(f"Saved CopulaGAN model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving CopulaGAN model: {str(e)}")
            raise

    def load(self, path: str):
        """
        Load a fitted CopulaGAN model from disk.

        Args:
            path: Path to load the model from.
        """
        try:
            self.model = CopulaGAN.load(path)
            self.logger.info(f"Loaded CopulaGAN model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading CopulaGAN model: {str(e)}")
            raise
