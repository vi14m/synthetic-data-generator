# import os
# import logging
# import pandas as pd
# from abc import ABC, abstractmethod
# from typing import Dict, List, Optional, Union, Any
# from sdv.metadata import Metadata

# class BaseGenerator(ABC):
#     """Abstract base class for all synthetic data generators."""
    
#     def __init__(self):
#         """Initialize the base generator."""
#         self.logger = logging.getLogger(__name__)
    
#     @abstractmethod
#     def configure(self, **kwargs):
#         """Configure the generator with specific parameters.
        
#         Args:
#             **kwargs: Configuration parameters
#         """
#         pass
    
#     @abstractmethod
#     def fit(self, data):
#         """Fit the generator to the input data.
        
#         Args:
#             data: Input data to learn from
            
#         Returns:
#             self: The fitted generator
#         """
#         pass
    
#     @abstractmethod
#     def generate(self, num_samples: int):
#         """Generate synthetic samples.
        
#         Args:
#             num_samples: Number of synthetic samples to generate
            
#         Returns:
#             Generated synthetic data
#         """
#         pass
    
#     @abstractmethod
#     def save(self, path: str):
#         """Save the fitted generator to disk.
        
#         Args:
#             path: Path to save the generator to
#         """
#         pass
    
#     @abstractmethod
#     def load(self, path: str):
#         """Load a fitted generator from disk.
        
#         Args:
#             path: Path to load the generator from
#         """
#         pass
    
#     def validate_data(self, data):
#         """Validate the input data format.
        
#         Args:
#             data: Input data to validate
            
#         Returns:
#             bool: True if the data is valid, False otherwise
#         """
#         if not isinstance(data, pd.DataFrame):
#             self.logger.error("Input data must be a pandas DataFrame")
#             return False
        
#         if data.empty:
#             self.logger.error("Input data is empty")
#             return False
        
#         return True
    
#     def get_metadata(self):
#         """Get metadata about the generator.
        
#         Returns:
#             Dict: Metadata about the generator
#         """
#         return {
#             "generator_type": self.__class__.__name__,
#             "configured": hasattr(self, "model") and self.model is not None
#         }

#     def _get_metadata(self, data: pd.DataFrame):
#         """Extract metadata from the input DataFrame.

#         Args:
#             data: Input DataFrame.

#         Returns:
#             dict: Metadata dictionary.
#         """
#         metadata = {
#             "columns": {},
#             "METADATA_SPEC_VERSION": "1.0"
#         }

#         for column in data.columns:
#             col_data = data[column]
#             if pd.api.types.is_numeric_dtype(col_data):
#                 if pd.api.types.is_integer_dtype(col_data):
#                     metadata["columns"][column] = {"type": "numerical", "subtype": "integer"}
#                 else:
#                     metadata["columns"][column] = {"type": "numerical", "subtype": "float"}
#             elif pd.api.types.is_datetime64_any_dtype(col_data):
#                 metadata["columns"][column] = {"type": "datetime"}
#             elif pd.api.types.is_categorical_dtype(col_data) or col_data.nunique() < len(col_data) * 0.1:
#                 metadata["columns"][column] = {"type": "categorical"}
#             else:
#                 metadata["columns"][column] = {"type": "text"}
#         metadata_obj = Metadata(metadata)
#         return metadata_obj


import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict
from sdv.metadata import SingleTableMetadata


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""
    
    def __init__(self):
        """Initialize the base generator."""
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def configure(self, **kwargs):
        """Configure the generator with specific parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
        pass
    
    @abstractmethod
    def fit(self, data):
        """Fit the generator to the input data.
        
        Args:
            data: Input data to learn from
            
        Returns:
            self: The fitted generator
        """
        pass
    
    @abstractmethod
    def generate(self, num_samples: int):
        """Generate synthetic samples.
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Generated synthetic data
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the fitted generator to disk.
        
        Args:
            path: Path to save the generator to
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load a fitted generator from disk.
        
        Args:
            path: Path to load the generator from
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if the data is valid, False otherwise
        """
        if not isinstance(data, pd.DataFrame):
            self.logger.error("Input data must be a pandas DataFrame")
            return False
        
        if data.empty:
            self.logger.error("Input data is empty")
            return False
        
        return True

    def get_metadata(self) -> Dict:
        """Get metadata about the generator.
        
        Returns:
            Dict: Metadata about the generator
        """
        return {
            "generator_type": self.__class__.__name__,
            "configured": hasattr(self, "model") and self.model is not None
        }

    def _get_metadata(self, data: pd.DataFrame) -> SingleTableMetadata:
        """Generate SDV SingleTableMetadata from the input DataFrame.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            SingleTableMetadata: Automatically inferred metadata object.
        """
        if not self.validate_data(data):
            raise ValueError("Invalid or empty DataFrame provided for metadata generation.")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        return metadata