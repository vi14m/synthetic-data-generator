import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Any

# Import base generator class
from .base_generator import BaseGenerator

class TimeGANGenerator(BaseGenerator):
    """TimeGAN generator for synthetic time-series data."""
    
    def __init__(self, **kwargs):
        """Initialize the TimeGAN generator.
        
        Args:
            **kwargs: Additional arguments to pass to the TimeGAN constructor
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_kwargs = kwargs
        self.seq_len = None
        self.n_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def configure(self, seq_len: int = 24, hidden_dim: int = 24, num_layers: int = 3, 
                 epochs: int = 200, batch_size: int = 128, **kwargs):
        """Configure the TimeGAN generator.
        
        Args:
            seq_len: Sequence length for time series data
            hidden_dim: Hidden dimension of the GAN
            num_layers: Number of layers in the GAN
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional configuration parameters
        """
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured TimeGAN with seq_len={seq_len}, hidden_dim={hidden_dim}, "
                        f"num_layers={num_layers}, epochs={epochs}, batch_size={batch_size}")
    
    def fit(self, data):
        """Fit the TimeGAN model to the input data.
        
        Args:
            data: Input time series data to learn from. Can be a DataFrame or numpy array.
            
        Returns:
            self: The fitted generator
        """
        try:
            # Convert data to appropriate format if needed
            if isinstance(data, pd.DataFrame):
                # Check if the index is a datetime index
                if pd.api.types.is_datetime64_any_dtype(data.index):
                    # Sort by datetime index
                    data = data.sort_index()
                
                # Convert to numpy array for TimeGAN
                data_array = data.values
            else:
                data_array = data
            
            # Ensure data is 3D: [batch, seq_len, features]
            if len(data_array.shape) == 2:
                # Reshape to [batch, seq_len, features]
                self.n_features = data_array.shape[1]
                
                # If seq_len not provided, use a default or infer from data
                if self.seq_len is None:
                    self.seq_len = min(24, data_array.shape[0] // 10)  # Default or 1/10 of data length
                
                # Prepare sequences
                sequences = self._prepare_sequences(data_array, self.seq_len)
            else:
                # Already in [batch, seq_len, features] format
                sequences = data_array
                self.seq_len = sequences.shape[1]
                self.n_features = sequences.shape[2]
            
            self.logger.info(f"Prepared {len(sequences)} sequences with shape [{self.seq_len}, {self.n_features}]")
            
            # Initialize and train TimeGAN model
            self._initialize_model()
            self._train_model(sequences)
            
            return self
            
        except ImportError:
            self.logger.error("Required packages not found. Please install them with 'pip install ydata-synthetic torch'")
            raise
        except Exception as e:
            self.logger.error(f"Error fitting TimeGAN model: {str(e)}")
            raise
    
    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic time series samples using the fitted TimeGAN model.
        
        Args:
            num_samples: Number of synthetic time series to generate
            
        Returns:
            DataFrame containing the generated synthetic time series
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            self.logger.info(f"Generating {num_samples} synthetic time series samples")
            
            # Generate synthetic sequences
            synthetic_sequences = self._generate_sequences(num_samples)
            
            # Convert to DataFrame
            synthetic_df = self._sequences_to_dataframe(synthetic_sequences)
            
            self.logger.info(f"Generated synthetic time series with shape {synthetic_df.shape}")
            
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic time series: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save the fitted TimeGAN model to disk.
        
        Args:
            path: Path to save the model to
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state dict
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'seq_len': self.seq_len,
                'n_features': self.n_features,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'model_kwargs': self.model_kwargs
            }, path)
            
            self.logger.info(f"Saved TimeGAN model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving TimeGAN model: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load a fitted TimeGAN model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            # Load model state dict
            checkpoint = torch.load(path, map_location=self.device)
            
            # Set model parameters
            self.seq_len = checkpoint['seq_len']
            self.n_features = checkpoint['n_features']
            self.hidden_dim = checkpoint['hidden_dim']
            self.num_layers = checkpoint['num_layers']
            self.model_kwargs = checkpoint['model_kwargs']
            
            # Initialize model
            self._initialize_model()
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"Loaded TimeGAN model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading TimeGAN model: {str(e)}")
            raise
    
    def _initialize_model(self):
        """Initialize the TimeGAN model."""
        try:
            # Import here to avoid dependency issues
            from ydata_synthetic.synthesizers import TimeGAN
            
            # Initialize TimeGAN model
            self.model = TimeGAN(
                seq_len=self.seq_len,
                n_features=self.n_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                **self.model_kwargs
            )
            
            self.logger.info("Initialized TimeGAN model")
        except ImportError:
            self.logger.error("ydata-synthetic package not found. Please install it with 'pip install ydata-synthetic'")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing TimeGAN model: {str(e)}")
            raise
    
    def _train_model(self, sequences):
        """Train the TimeGAN model on the prepared sequences.
        
        Args:
            sequences: Prepared time series sequences
        """
        self.logger.info(f"Training TimeGAN model for {self.epochs} epochs with batch size {self.batch_size}")
        
        # Train the model
        self.model.train(
            sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True
        )
        
        self.logger.info("TimeGAN model training completed")
    
    def _generate_sequences(self, num_samples):
        """Generate synthetic sequences using the trained model.
        
        Args:
            num_samples: Number of sequences to generate
            
        Returns:
            Numpy array of generated sequences
        """
        # Generate synthetic sequences
        synthetic_sequences = self.model.sample(num_samples)
        
        return synthetic_sequences
    
    def _prepare_sequences(self, data, seq_len):
        """Prepare sequences from time series data for TimeGAN.
        
        Args:
            data: Time series data as numpy array
            seq_len: Sequence length
            
        Returns:
            Numpy array of sequences
        """
        n_samples = data.shape[0] - seq_len + 1
        sequences = np.zeros((n_samples, seq_len, self.n_features))
        
        for i in range(n_samples):
            sequences[i] = data[i:i+seq_len]
        
        return sequences
    
    def _sequences_to_dataframe(self, sequences):
        """Convert generated sequences to a DataFrame.
        
        Args:
            sequences: Generated sequences as numpy array
            
        Returns:
            DataFrame with generated time series
        """
        # Reshape to 2D if needed
        if len(sequences.shape) == 3:
            # For simplicity, we'll use the first sequence from each sample
            sequences_2d = sequences[:, 0, :]
        else:
            sequences_2d = sequences
        
        # Create DataFrame
        columns = [f'feature_{i}' for i in range(sequences_2d.shape[1])]
        df = pd.DataFrame(sequences_2d, columns=columns)
        
        # Add a time index
        df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        df.set_index('time', inplace=True)
        
        return df