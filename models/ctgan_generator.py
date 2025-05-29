import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sdv.single_table import CTGANSynthesizer as CTGAN
from sdv.metadata import SingleTableMetadata

from .base_generator import BaseGenerator


class CTGANGenerator(BaseGenerator):
    """CTGAN (Conditional Tabular GAN) generator for synthetic tabular data."""
    
    def __init__(self, metadata: Optional[SingleTableMetadata] = None, **kwargs):
        """Initialize the CTGAN generator."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_kwargs = kwargs
        self.discrete_columns = []
        self.continuous_columns = []
        self.metadata = metadata
    
    def configure(self, epochs: int = 300, batch_size: int = 500, embedding_dim: int = 128, **kwargs):
        """Configure the CTGAN generator."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured CTGAN with epochs={epochs}, batch_size={batch_size}, embedding_dim={embedding_dim}")
    
    def fit(self, data: pd.DataFrame):
        """Fit the CTGAN model to the input data."""
        if not self.validate_data(data):
            raise ValueError("Invalid data provided for fitting.")
        
        try:
            self.logger.info(f"Fitting CTGAN model on data with shape {data.shape}")
            self._identify_column_types(data)

            # Create metadata if not provided
            metadata_obj = self.metadata if self.metadata is not None else self._get_metadata(data)
            
            # Ensure epochs attribute is set
            if not hasattr(self, 'epochs'):
                self.logger.warning("'epochs' attribute not set. Using default value of 300.")
                self.epochs = 300
                
            if not hasattr(self, 'batch_size'):
                self.batch_size = 500
                
            if not hasattr(self, 'embedding_dim'):
                self.embedding_dim = 128

            # Initialize model with metadata
            self.model = CTGAN(
                metadata=metadata_obj,
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                enforce_min_max_values=True,
                enforce_rounding=True,  # Enable rounding for integer columns
                verbose=True,
                **self.model_kwargs
            )

            self.logger.info(f"Training CTGAN with epochs={self.epochs}, batch_size={self.batch_size}")
            self.model.fit(data)
            self.logger.info("CTGAN model fitting completed")
            return self

        except Exception as e:
            self.logger.error(f"Error fitting CTGAN model: {str(e)}")
            raise
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic samples using the fitted CTGAN model."""
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
        """Save the fitted CTGAN model to disk."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            self.logger.info(f"Saved CTGAN model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving CTGAN model: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load a fitted CTGAN model from disk."""
        try:
            self.model = CTGAN.load(path)
            self.logger.info(f"Loaded CTGAN model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading CTGAN model: {str(e)}")
            raise
    
    def _identify_column_types(self, data: pd.DataFrame):
        """Identify discrete and continuous columns in the data."""
        self.discrete_columns = []
        self.continuous_columns = []
        
        for column in data.columns:
            if data[column].dtype == 'object' or data[column].dtype == 'category' or \
               (data[column].dtype == 'int64' and len(data[column].unique()) < 50):
                self.discrete_columns.append(column)
            else:
                self.continuous_columns.append(column)
        
        self.logger.info(f"Identified {len(self.discrete_columns)} discrete columns and {len(self.continuous_columns)} continuous columns")
