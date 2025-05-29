import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sdv.single_table import TVAESynthesizer as TVAE
from .base_generator import BaseGenerator


class TVAEGenerator(BaseGenerator):
    """TVAE (Tabular Variational Autoencoder) generator for synthetic tabular data."""
    
    def __init__(self, metadata=None, **kwargs):
        """Initialize the TVAE generator."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_kwargs = kwargs
        self.discrete_columns = []
        self.continuous_columns = []
        self.metadata = metadata
        self.epochs = 300
        self.batch_size = 500
        self.embedding_dim = 128
    
    def configure(self, epochs: int = 300, batch_size: int = 500, embedding_dim: int = 128, **kwargs):
        """Configure the TVAE generator."""
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured TVAE with epochs={epochs}, batch_size={batch_size}, embedding_dim={embedding_dim}")
    
    def fit(self, data: pd.DataFrame):
        """Fit the TVAE model to the input data."""
        try:
            self._identify_column_types(data)
            
            metadata_obj = self.metadata if self.metadata is not None else self._get_metadata(data)
            
            self.model = TVAE(
                metadata=metadata_obj,
                enforce_min_max_values=True,
                enforce_rounding=False,
                epochs=self.epochs,
                batch_size=self.batch_size,
                embedding_dim=self.embedding_dim,
                verbose=True,
                **self.model_kwargs
            )
            
            self.logger.info(f"Fitting TVAE model on data with shape {data.shape}")
            self.model.fit(data)
            self.logger.info("TVAE model fitting completed")
            
            return self
        
        except ImportError:
            self.logger.error("SDV package not found. Please install it with 'pip install sdv'")
            raise
        except Exception as e:
            self.logger.error(f"Error fitting TVAE model: {str(e)}")
            raise
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic samples using the fitted TVAE model."""
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
        """Save the fitted TVAE model to disk."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            self.logger.info(f"Saved TVAE model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving TVAE model: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load a fitted TVAE model from disk."""
        try:
            self.model = TVAE.load(path)
            self.logger.info(f"Loaded TVAE model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading TVAE model: {str(e)}")
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
