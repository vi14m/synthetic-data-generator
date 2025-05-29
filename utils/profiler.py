import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

class DataProfiler:
    """Data profiler for analyzing and profiling input data."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize the data profiler.
        
        Args:
            data: Input DataFrame to profile
        """
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.profile = None
        self.logger.info(f"Initialized data profiler for data with shape {data.shape}")
    
    def generate_profile(self) -> Dict[str, Any]:
        """Generate a comprehensive profile of the input data.
        
        Returns:
            Dictionary containing the data profile
        """
        self.logger.info("Generating data profile")
        
        profile = {}
        
        # Basic statistics
        profile['row_count'] = len(self.data)
        profile['column_count'] = len(self.data.columns)
        profile['missing_count'] = self.data.isna().sum().sum()
        profile['missing_percentage'] = (profile['missing_count'] / (profile['row_count'] * profile['column_count'])) * 100
        
        # Column types
        profile['column_types'] = self._get_column_types()
        
        # Column statistics
        profile['column_stats'] = self._get_column_statistics()
        
        # Correlation matrix
        profile['correlation_matrix'] = self._get_correlation_matrix()
        
        # Data quality issues
        profile['quality_issues'] = self._check_data_quality()
        
        # Data schema
        profile['schema'] = self._infer_schema()
        
        self.profile = profile
        self.logger.info("Data profile generated successfully")
        
        return profile
    
    def generate_ydata_profile(self, output_path: Optional[str] = None) -> str:
        """Generate a detailed profile using ydata-profiling (formerly pandas-profiling).
        
        Args:
            output_path: Optional path to save the profile report
            
        Returns:
            Path to the saved profile report or HTML string if no path provided
        """
        try:
            from ydata_profiling import ProfileReport
            
            self.logger.info("Generating ydata profile report")
            
            # Generate profile report
            profile = ProfileReport(self.data, title="Data Profile Report")
            
            # Save or return the report
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                profile.to_file(output_path)
                self.logger.info(f"Saved ydata profile report to {output_path}")
                return output_path
            else:
                html_report = profile.to_html()
                self.logger.info("Generated ydata profile report as HTML string")
                return html_report
                
        except ImportError:
            self.logger.warning("ydata-profiling package not found. Please install it with 'pip install ydata-profiling'")
            return "ydata-profiling package not installed"
        except Exception as e:
            self.logger.error(f"Error generating ydata profile report: {str(e)}")
            return str(e)
    
    def _get_column_types(self) -> Dict[str, str]:
        """Get the data types of each column.
        
        Returns:
            Dictionary mapping column names to their data types
        """
        column_types = {}
        
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                if pd.api.types.is_integer_dtype(self.data[column]):
                    column_types[column] = 'integer'
                else:
                    column_types[column] = 'float'
            elif pd.api.types.is_datetime64_any_dtype(self.data[column]):
                column_types[column] = 'datetime'
            elif pd.api.types.is_bool_dtype(self.data[column]):
                column_types[column] = 'boolean'
            elif pd.api.types.is_categorical_dtype(self.data[column]):
                column_types[column] = 'categorical'
            else:
                # Check if it's a categorical column based on cardinality
                if len(self.data[column].unique()) < min(50, len(self.data) * 0.1):
                    column_types[column] = 'categorical'
                else:
                    column_types[column] = 'text'
        
        return column_types
    
    def _get_column_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each column.
        
        Returns:
            Dictionary mapping column names to their statistics
        """
        column_stats = {}
        
        for column in self.data.columns:
            stats = {}
            
            # Common statistics for all types
            stats['missing_count'] = self.data[column].isna().sum()
            stats['missing_percentage'] = (stats['missing_count'] / len(self.data)) * 100
            stats['unique_count'] = self.data[column].nunique()
            stats['unique_percentage'] = (stats['unique_count'] / len(self.data)) * 100
            
            # Type-specific statistics
            if pd.api.types.is_numeric_dtype(self.data[column]):
                stats['min'] = float(self.data[column].min())
                stats['max'] = float(self.data[column].max())
                stats['mean'] = float(self.data[column].mean())
                stats['median'] = float(self.data[column].median())
                stats['std'] = float(self.data[column].std())
                stats['skewness'] = float(self.data[column].skew())
                stats['kurtosis'] = float(self.data[column].kurtosis())
                
                # Check for outliers using IQR method
                q1 = float(self.data[column].quantile(0.25))
                q3 = float(self.data[column].quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)][column]
                stats['outlier_count'] = len(outliers)
                stats['outlier_percentage'] = (len(outliers) / len(self.data)) * 100
                
            elif pd.api.types.is_datetime64_any_dtype(self.data[column]):
                stats['min'] = self.data[column].min().strftime('%Y-%m-%d %H:%M:%S')
                stats['max'] = self.data[column].max().strftime('%Y-%m-%d %H:%M:%S')
                stats['range_days'] = (self.data[column].max() - self.data[column].min()).days
                
            else:  # Categorical or text
                value_counts = self.data[column].value_counts()
                stats['most_common'] = value_counts.index[0] if not value_counts.empty else None
                stats['most_common_count'] = int(value_counts.iloc[0]) if not value_counts.empty else 0
                stats['most_common_percentage'] = (stats['most_common_count'] / len(self.data)) * 100
                
                # For text columns, add text-specific statistics
                if self._get_column_types()[column] == 'text':
                    # Calculate average length of text
                    stats['avg_length'] = float(self.data[column].astype(str).str.len().mean())
                    stats['max_length'] = int(self.data[column].astype(str).str.len().max())
            
            column_stats[column] = stats
        
        return column_stats
    
    def _get_correlation_matrix(self) -> pd.DataFrame:
        """Get the correlation matrix for numeric columns.
        
        Returns:
            DataFrame containing the correlation matrix
        """
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            self.logger.warning("No numeric columns found for correlation matrix")
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        return correlation_matrix
    
    def _check_data_quality(self) -> Dict[str, List[Dict[str, Any]]]:
        """Check for data quality issues.
        
        Returns:
            Dictionary of data quality issues by category
        """
        quality_issues = {
            'missing_data': [],
            'outliers': [],
            'inconsistencies': [],
            'duplicates': []
        }
        
        # Check for columns with high missing values
        for column in self.data.columns:
            missing_percentage = (self.data[column].isna().sum() / len(self.data)) * 100
            if missing_percentage > 5:
                quality_issues['missing_data'].append({
                    'column': column,
                    'missing_percentage': missing_percentage
                })
        
        # Check for outliers in numeric columns
        for column in self.data.select_dtypes(include=['number']).columns:
            q1 = self.data[column].quantile(0.25)
            q3 = self.data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_percentage = ((self.data[column] < lower_bound) | (self.data[column] > upper_bound)).mean() * 100
            
            if outlier_percentage > 1:
                quality_issues['outliers'].append({
                    'column': column,
                    'outlier_percentage': outlier_percentage
                })
        
        # Check for inconsistencies in categorical columns
        for column, col_type in self._get_column_types().items():
            if col_type == 'categorical':
                # Check for rare categories
                value_counts = self.data[column].value_counts(normalize=True) * 100
                rare_categories = value_counts[value_counts < 1].index.tolist()
                
                if rare_categories:
                    quality_issues['inconsistencies'].append({
                        'column': column,
                        'issue': 'rare_categories',
                        'categories': rare_categories,
                        'count': len(rare_categories)
                    })
        
        # Check for duplicates
        duplicate_rows = self.data.duplicated().sum()
        if duplicate_rows > 0:
            quality_issues['duplicates'].append({
                'duplicate_rows': int(duplicate_rows),
                'duplicate_percentage': (duplicate_rows / len(self.data)) * 100
            })
        
        return quality_issues
    
    def _infer_schema(self) -> Dict[str, Dict[str, Any]]:
        """Infer the schema of the data.
        
        Returns:
            Dictionary containing the inferred schema
        """
        schema = {}
        column_types = self._get_column_types()
        
        for column, col_type in column_types.items():
            column_schema = {
                'type': col_type,
                'nullable': self.data[column].isna().any()
            }
            
            # Add type-specific schema information
            if col_type == 'integer' or col_type == 'float':
                column_schema['min'] = float(self.data[column].min())
                column_schema['max'] = float(self.data[column].max())
            elif col_type == 'categorical':
                column_schema['categories'] = self.data[column].dropna().unique().tolist()
            elif col_type == 'datetime':
                column_schema['format'] = 'infer'  # Could be improved with format detection
            
            schema[column] = column_schema
        
        return schema
    
    def suggest_transformations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Suggest data transformations based on the profile.
        
        Returns:
            Dictionary of suggested transformations by column
        """
        if self.profile is None:
            self.generate_profile()
        
        transformations = {}
        
        for column, stats in self.profile['column_stats'].items():
            column_transformations = []
            col_type = self.profile['column_types'][column]
            
            # Suggestions for missing values
            if stats['missing_percentage'] > 0:
                if col_type in ['integer', 'float']:
                    column_transformations.append({
                        'type': 'impute_missing',
                        'method': 'mean',
                        'reason': f"Column has {stats['missing_percentage']:.2f}% missing values"
                    })
                elif col_type == 'categorical':
                    column_transformations.append({
                        'type': 'impute_missing',
                        'method': 'mode',
                        'reason': f"Column has {stats['missing_percentage']:.2f}% missing values"
                    })
            
            # Suggestions for numeric columns
            if col_type in ['integer', 'float']:
                # Check for skewness
                if 'skewness' in stats and abs(stats['skewness']) > 1:
                    column_transformations.append({
                        'type': 'transform',
                        'method': 'log' if stats['skewness'] > 0 else 'square',
                        'reason': f"Column is {'positively' if stats['skewness'] > 0 else 'negatively'} skewed (skewness: {stats['skewness']:.2f})"
                    })
                
                # Check for outliers
                if 'outlier_percentage' in stats and stats['outlier_percentage'] > 5:
                    column_transformations.append({
                        'type': 'handle_outliers',
                        'method': 'clip',
                        'reason': f"Column has {stats['outlier_percentage']:.2f}% outliers"
                    })
                
                # Suggest normalization
                column_transformations.append({
                    'type': 'normalize',
                    'method': 'standard' if stats['min'] < 0 or stats['max'] > 10 else 'minmax',
                    'reason': "Normalize numeric features for better model performance"
                })
            
            # Suggestions for categorical columns
            elif col_type == 'categorical':
                # Check for high cardinality
                if stats['unique_count'] > 10:
                    column_transformations.append({
                        'type': 'encode',
                        'method': 'target' if stats['unique_count'] < 50 else 'hash',
                        'reason': f"Column has high cardinality ({stats['unique_count']} unique values)"
                    })
                else:
                    column_transformations.append({
                        'type': 'encode',
                        'method': 'onehot',
                        'reason': f"Column has low cardinality ({stats['unique_count']} unique values)"
                    })
            
            # Suggestions for text columns
            elif col_type == 'text':
                column_transformations.append({
                    'type': 'text_process',
                    'method': 'tfidf',
                    'reason': "Convert text to numerical features"
                })
            
            # Suggestions for datetime columns
            elif col_type == 'datetime':
                column_transformations.append({
                    'type': 'extract_features',
                    'method': 'datetime_components',
                    'reason': "Extract year, month, day, etc. from datetime"
                })
            
            if column_transformations:
                transformations[column] = column_transformations
        
        return transformations
    
    def detect_data_type(self) -> str:
        """Detect the type of data (tabular, time-series, text, image).
        
        Returns:
            String indicating the detected data type
        """
        # Check if it's time series data
        if self._is_timeseries():
            return 'timeseries'
        
        # Check if it's primarily text data
        text_columns = [col for col, col_type in self._get_column_types().items() if col_type == 'text']
        if len(text_columns) > len(self.data.columns) * 0.5:
            return 'text'
        
        # Default to tabular
        return 'tabular'
    
    def _is_timeseries(self) -> bool:
        """Check if the data appears to be time series data.
        
        Returns:
            Boolean indicating if the data appears to be time series
        """
        # Check for datetime index
        if pd.api.types.is_datetime64_any_dtype(self.data.index):
            return True
        
        # Check for datetime columns
        datetime_cols = [col for col in self.data.columns if pd.api.types.is_datetime64_any_dtype(self.data[col])]
        if len(datetime_cols) > 0:
            return True
        
        # Check for column names that suggest time series
        time_related_cols = [col for col in self.data.columns if any(term in col.lower() for term in 
                                                               ['time', 'date', 'year', 'month', 'day', 'hour', 'minute', 'second'])]
        if len(time_related_cols) > 0:
            return True
        
        return False