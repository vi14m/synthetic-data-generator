import os
import logging
from typing import Dict, Any, Optional, Union

# Import generators
from .ctgan_generator import CTGANGenerator
from .tvae_generator import TVAEGenerator
from .copulagan_generator import CopulaGANGenerator
from .timegan_generator import TimeGANGenerator
from .text_generator import TextGenerator
from .image_generator import ImageGenerator

class GeneratorFactory:
    """Factory class for creating appropriate synthetic data generators based on data type."""
    
    def __init__(self):
        """Initialize the generator factory."""
        self.logger = logging.getLogger(__name__)
        self.generators = {
            # Tabular generators
            'ctgan': CTGANGenerator,
            'tvae': TVAEGenerator,
            'copulagan': CopulaGANGenerator,
            
            # Time series generator
            'timegan': TimeGANGenerator,
            
            # Text generator
            'text': TextGenerator,
            
            # Image generator
            'image': ImageGenerator
        }
    
    def get_generator(self, generator_type: str, **kwargs):
        """Get the appropriate generator based on the generator type.
        
        Args:
            generator_type: Type of generator to create
            **kwargs: Additional arguments to pass to the generator constructor
            
        Returns:
            An instance of the appropriate generator
            
        Raises:
            ValueError: If the generator type is not supported
        """
        if generator_type not in self.generators:
            raise ValueError(f"Unsupported generator type: {generator_type}")
        
        self.logger.info(f"Creating generator of type: {generator_type}")
        return self.generators[generator_type](**kwargs)
    
    def get_recommended_generator(self, data, data_type: str = None):
        """Automatically recommend the best generator based on the data characteristics.
        
        Args:
            data: The input data to analyze
            data_type: Optional explicit data type (tabular, timeseries, text, image)
            
        Returns:
            A tuple of (generator_type, generator_instance)
        """
        if data_type is None:
            # Try to infer data type
            data_type = self._infer_data_type(data)
        
        self.logger.info(f"Inferred data type: {data_type}")
        
        if data_type == 'tabular':
            # For tabular data, recommend CTGAN as default
            return 'ctgan', self.get_generator('ctgan')
        elif data_type == 'timeseries':
            return 'timegan', self.get_generator('timegan')
        elif data_type == 'text':
            return 'text', self.get_generator('text')
        elif data_type == 'image':
            return 'image', self.get_generator('image')
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _infer_data_type(self, data):
        """Infer the data type from the data structure.
        
        Args:
            data: The input data to analyze
            
        Returns:
            String indicating the inferred data type
        """
        import pandas as pd
        import numpy as np
        from PIL import Image
        
        if isinstance(data, pd.DataFrame):
            # Check if it's time series data
            if self._is_timeseries(data):
                return 'timeseries'
            else:
                return 'tabular'
        elif isinstance(data, str) or (isinstance(data, list) and all(isinstance(x, str) for x in data)):
            return 'text'
        elif isinstance(data, Image.Image) or isinstance(data, np.ndarray) and len(data.shape) >= 2:
            return 'image'
        else:
            # Default to tabular
            return 'tabular'
    
    def _is_timeseries(self, df):
        """Check if a DataFrame appears to be time series data.
        
        Args:
            df: Pandas DataFrame to check
            
        Returns:
            Boolean indicating if the data appears to be time series
        """
        # Check for datetime index
        if pd.api.types.is_datetime64_any_dtype(df.index):
            return True
        
        # Check for datetime columns
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if len(datetime_cols) > 0:
            return True
        
        # Check for column names that suggest time series
        time_related_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                           ['time', 'date', 'year', 'month', 'day', 'hour', 'minute', 'second'])]
        if len(time_related_cols) > 0:
            return True
        
        return False