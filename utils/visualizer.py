import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from io import BytesIO
import base64

class DataVisualizer:
    """Utility class for generating visualizations for original and synthetic data."""
    
    def __init__(self, original_data: pd.DataFrame, synthetic_data: Optional[pd.DataFrame] = None):
        """Initialize the data visualizer.
        
        Args:
            original_data: Original input DataFrame
            synthetic_data: Optional synthetic DataFrame for comparison
        """
        self.logger = logging.getLogger(__name__)
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.logger.info(f"Initialized data visualizer for data with shape {original_data.shape}")
    
    def update_synthetic_data(self, synthetic_data: pd.DataFrame) -> None:
        """Update the synthetic data reference.
        
        Args:
            synthetic_data: New synthetic DataFrame for comparison
        """
        self.synthetic_data = synthetic_data
        self.logger.info(f"Updated synthetic data with shape {synthetic_data.shape}")
    
    def get_distribution_plots(self, columns: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 8)) -> str:
        """Generate distribution plots for numeric columns.
        
        Args:
            columns: List of columns to visualize (defaults to all numeric columns)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the plot image
        """
        # Select columns to visualize
        if columns is None:
            columns = self.original_data.select_dtypes(include=['number']).columns.tolist()
            if len(columns) > 10:  # Limit to top 10 columns if there are too many
                self.logger.info(f"Limiting distribution plots to first 10 of {len(columns)} numeric columns")
                columns = columns[:10]
        
        # Create figure
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Generate distribution plots
        for i, column in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                # Plot original data
                sns.histplot(self.original_data[column], kde=True, ax=ax, color='blue', alpha=0.5, label='Original')
                
                # Plot synthetic data if available
                if self.synthetic_data is not None and column in self.synthetic_data.columns:
                    sns.histplot(self.synthetic_data[column], kde=True, ax=ax, color='red', alpha=0.5, label='Synthetic')
                
                ax.set_title(f'Distribution of {column}')
                ax.legend()
                ax.set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def get_distribution_plot(self, column: str, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Generate distribution plot for a single column.
        
        Args:
            column: Column name to visualize
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure object
        """
        # Validate column exists
        if column not in self.original_data.columns:
            self.logger.error(f"Column '{column}' not found in original data")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Column '{column}' not found in original data", 
                    ha='center', va='center', fontsize=12)
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine plot type based on data type
        if pd.api.types.is_numeric_dtype(self.original_data[column]):
            # Numeric column - use histogram with KDE
            sns.histplot(self.original_data[column], kde=True, ax=ax, 
                         color='blue', alpha=0.5, label='Original')
            
            # Plot synthetic data if available
            if self.synthetic_data is not None and column in self.synthetic_data.columns:
                sns.histplot(self.synthetic_data[column], kde=True, ax=ax, 
                             color='red', alpha=0.5, label='Synthetic')
        else:
            # Categorical column - use bar plot
            # Get value counts for original data (limit to top 10 categories)
            orig_counts = self.original_data[column].value_counts().nlargest(10)
            
            # Get value counts for synthetic data if available
            if self.synthetic_data is not None and column in self.synthetic_data.columns:
                syn_counts = self.synthetic_data[column].value_counts()
                # Align with original categories
                syn_counts = syn_counts.reindex(orig_counts.index, fill_value=0)
                
                # Plot side by side bars
                x = np.arange(len(orig_counts.index))
                width = 0.35
                ax.bar(x - width/2, orig_counts.values, width, label='Original', color='blue', alpha=0.7)
                ax.bar(x + width/2, syn_counts.values, width, label='Synthetic', color='red', alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(orig_counts.index, rotation=45, ha='right')
            else:
                # Plot only original data
                orig_counts.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
        
        ax.set_title(f'Distribution of {column}')
        ax.legend()
        ax.set_ylabel('Frequency' if pd.api.types.is_numeric_dtype(self.original_data[column]) else 'Count')
        plt.tight_layout()
        
        return fig
        
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Base64 encoded string of the figure
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str
    
    def get_correlation_heatmap(self, figsize: Tuple[int, int] = (10, 8)) -> Dict[str, str]:
        """Generate correlation heatmaps for original and synthetic data.
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Dictionary with base64 encoded strings of the heatmap images
        """
        result = {}
        
        # Get numeric columns
        numeric_cols = self.original_data.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            self.logger.warning("No numeric columns found for correlation heatmap")
            return {"error": "No numeric columns found"}
        
        # Original data correlation heatmap
        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = self.original_data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
        ax.set_title("Original Data Correlation Heatmap")
        plt.tight_layout()
        result["original"] = self._fig_to_base64(fig)
        
        # Synthetic data correlation heatmap (if available)
        if self.synthetic_data is not None:
            fig, ax = plt.subplots(figsize=figsize)
            synthetic_numeric_cols = [col for col in numeric_cols if col in self.synthetic_data.columns]
            if synthetic_numeric_cols:
                corr_matrix = self.synthetic_data[synthetic_numeric_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
                ax.set_title("Synthetic Data Correlation Heatmap")
                plt.tight_layout()
                result["synthetic"] = self._fig_to_base64(fig)
            else:
                result["synthetic"] = "No matching numeric columns found in synthetic data"
        
        return result
    
    def get_pairplot(self, columns: Optional[List[str]] = None, n_samples: int = 1000, figsize: Tuple[int, int] = (12, 12)) -> str:
        """Generate pairplot for selected columns.
        
        Args:
            columns: List of columns to include (defaults to 5 numeric columns)
            n_samples: Number of samples to use (for performance)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the pairplot image
        """
        # Select columns to visualize
        if columns is None:
            numeric_cols = self.original_data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) > 5:  # Limit to 5 columns for readability
                columns = numeric_cols[:5]
            else:
                columns = numeric_cols
        
        if not columns:
            self.logger.warning("No suitable columns found for pairplot")
            return "No suitable columns found"
        
        # Sample data for performance
        orig_sample = self.original_data[columns].sample(min(n_samples, len(self.original_data)))
        orig_sample['Dataset'] = 'Original'
        
        if self.synthetic_data is not None:
            # Ensure all columns exist in synthetic data
            valid_columns = [col for col in columns if col in self.synthetic_data.columns]
            if not valid_columns:
                self.logger.warning("No matching columns found in synthetic data for pairplot")
                return self._generate_single_pairplot(orig_sample, columns, figsize)
            
            syn_sample = self.synthetic_data[valid_columns].sample(min(n_samples, len(self.synthetic_data)))
            syn_sample['Dataset'] = 'Synthetic'
            
            # Combine samples
            combined = pd.concat([orig_sample[valid_columns + ['Dataset']], 
                                 syn_sample[valid_columns + ['Dataset']]])
            
            # Generate pairplot
            plt.figure(figsize=figsize)
            g = sns.pairplot(combined, hue='Dataset', palette=['blue', 'red'], 
                             plot_kws={'alpha': 0.5}, diag_kind='kde')
            g.fig.suptitle('Pairwise Relationships: Original vs Synthetic', y=1.02)
            plt.tight_layout()
            return self._fig_to_base64(g.fig)
        else:
            return self._generate_single_pairplot(orig_sample, columns, figsize)
    
    def _generate_single_pairplot(self, data: pd.DataFrame, columns: List[str], figsize: Tuple[int, int]) -> str:
        """Generate pairplot for a single dataset.
        
        Args:
            data: DataFrame to visualize
            columns: List of columns to include
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the pairplot image
        """
        plt.figure(figsize=figsize)
        g = sns.pairplot(data[columns], diag_kind='kde')
        g.fig.suptitle('Pairwise Relationships: Original Data', y=1.02)
        plt.tight_layout()
        return self._fig_to_base64(g.fig)
    
    def get_categorical_plots(self, columns: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 8)) -> str:
        """Generate bar plots for categorical columns.
        
        Args:
            columns: List of categorical columns to visualize
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the plot image
        """
        # Identify categorical columns if not specified
        if columns is None:
            columns = []
            for col in self.original_data.columns:
                if self.original_data[col].dtype == 'object' or \
                   pd.api.types.is_categorical_dtype(self.original_data[col]) or \
                   (self.original_data[col].nunique() < min(20, len(self.original_data) * 0.1)):
                    columns.append(col)
            
            if len(columns) > 6:  # Limit to 6 columns for readability
                columns = columns[:6]
        
        if not columns:
            self.logger.warning("No categorical columns found for bar plots")
            return "No categorical columns found"
        
        # Create figure
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Generate bar plots
        for i, column in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                # Get value counts for original data
                orig_counts = self.original_data[column].value_counts().nlargest(10)
                
                # Get value counts for synthetic data if available
                if self.synthetic_data is not None and column in self.synthetic_data.columns:
                    syn_counts = self.synthetic_data[column].value_counts()
                    # Align with original categories
                    syn_counts = syn_counts.reindex(orig_counts.index, fill_value=0)
                    
                    # Plot side by side bars
                    x = np.arange(len(orig_counts.index))
                    width = 0.35
                    ax.bar(x - width/2, orig_counts.values, width, label='Original', color='blue', alpha=0.7)
                    ax.bar(x + width/2, syn_counts.values, width, label='Synthetic', color='red', alpha=0.7)
                    ax.set_xticks(x)
                    ax.set_xticklabels(orig_counts.index, rotation=45, ha='right')
                else:
                    # Plot only original data
                    orig_counts.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
                
                ax.set_title(f'Distribution of {column}')
                ax.legend()
                ax.set_ylabel('Count')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def get_time_series_plot(self, time_column: str, value_columns: List[str], figsize: Tuple[int, int] = (12, 8)) -> str:
        """Generate time series plot.
        
        Args:
            time_column: Column containing time/date information
            value_columns: List of columns with values to plot over time
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the plot image
        """
        # Validate inputs
        if time_column not in self.original_data.columns:
            self.logger.error(f"Time column '{time_column}' not found in data")
            return f"Time column '{time_column}' not found in data"
        
        valid_value_columns = [col for col in value_columns if col in self.original_data.columns]
        if not valid_value_columns:
            self.logger.error(f"None of the specified value columns found in data")
            return f"None of the specified value columns found in data"
        
        # Ensure time column is datetime
        orig_data = self.original_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(orig_data[time_column]):
            try:
                orig_data[time_column] = pd.to_datetime(orig_data[time_column])
            except Exception as e:
                self.logger.error(f"Failed to convert '{time_column}' to datetime: {str(e)}")
                return f"Failed to convert '{time_column}' to datetime: {str(e)}"
        
        # Sort by time
        orig_data = orig_data.sort_values(by=time_column)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot original data
        for column in valid_value_columns:
            ax.plot(orig_data[time_column], orig_data[column], label=f'Original - {column}', alpha=0.7)
        
        # Plot synthetic data if available
        if self.synthetic_data is not None:
            syn_data = self.synthetic_data.copy()
            
            # Check if time column exists in synthetic data
            if time_column in syn_data.columns:
                # Ensure time column is datetime
                if not pd.api.types.is_datetime64_any_dtype(syn_data[time_column]):
                    try:
                        syn_data[time_column] = pd.to_datetime(syn_data[time_column])
                    except Exception as e:
                        self.logger.warning(f"Failed to convert synthetic '{time_column}' to datetime: {str(e)}")
                        return self._fig_to_base64(fig)  # Return original data plot only
                
                # Sort by time
                syn_data = syn_data.sort_values(by=time_column)
                
                # Plot synthetic data
                for column in valid_value_columns:
                    if column in syn_data.columns:
                        ax.plot(syn_data[time_column], syn_data[column], 
                                label=f'Synthetic - {column}', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Time Series Plot')
        ax.set_xlabel(time_column)
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def get_similarity_plot(self, metric_name: str, column_metrics: Dict[str, float], figsize: Tuple[int, int] = (10, 6)) -> str:
        """Generate bar plot showing similarity metrics across columns.
        
        Args:
            metric_name: Name of the similarity metric
            column_metrics: Dictionary mapping column names to metric values
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the plot image
        """
        if not column_metrics:
            self.logger.warning(f"No {metric_name} metrics provided")
            return f"No {metric_name} metrics provided"
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort columns by metric value
        sorted_items = sorted(column_metrics.items(), key=lambda x: x[1])
        columns = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Generate bar plot
        bars = ax.barh(columns, values, color='skyblue')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{values[i]:.3f}', va='center')
        
        ax.set_title(f'{metric_name} by Column')
        ax.set_xlabel(metric_name)
        ax.set_xlim(0, max(values) * 1.1)  # Add some padding
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def get_privacy_risk_plot(self, risk_scores: Dict[str, float], figsize: Tuple[int, int] = (8, 8)) -> str:
        """Generate gauge chart showing privacy risk levels.
        
        Args:
            risk_scores: Dictionary mapping risk types to scores (0-1)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Base64 encoded string of the plot image
        """
        if not risk_scores:
            self.logger.warning("No privacy risk scores provided")
            return "No privacy risk scores provided"
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        
        # Define risk levels and colors
        risk_levels = ['Low', 'Medium', 'High']
        colors = ['green', 'orange', 'red']
        bounds = [0, 0.33, 0.66, 1]
        
        # Create background for risk levels
        theta = np.linspace(0, 180, 100) * np.pi / 180
        for i, (level, color) in enumerate(zip(risk_levels, colors)):
            lower = bounds[i]
            upper = bounds[i+1]
            r_lower = np.ones_like(theta) * lower
            r_upper = np.ones_like(theta) * upper
            ax.fill_between(theta, r_lower, r_upper, color=color, alpha=0.2)
        
        # Plot each risk score
        width = 180 / len(risk_scores)
        for i, (risk_type, score) in enumerate(risk_scores.items()):
            # Calculate position
            pos = i * width
            theta_pos = pos * np.pi / 180
            
            # Determine color based on score
            if score <= 0.33:
                color = colors[0]
            elif score <= 0.66:
                color = colors[1]
            else:
                color = colors[2]
            
            # Plot needle
            ax.plot([theta_pos, theta_pos], [0, score], color=color, linewidth=3)
            ax.plot([theta_pos], [score], marker='o', markersize=8, color=color)
            
            # Add label
            label_r = 1.1
            ax.text(theta_pos, label_r, risk_type, 
                    ha='center', va='center', rotation=pos-90, fontsize=10)
            
            # Add score
            score_r = score - 0.15
            ax.text(theta_pos, max(0.1, score_r), f'{score:.2f}', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Configure plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_rlim(0, 1.2)
        ax.set_rticks([0.33, 0.66, 1])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.grid(True, alpha=0.3)
        
        # Set title
        ax.set_title('Privacy Risk Assessment', y=1.1, fontsize=14)
        
        # Adjust layout and convert to base64
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded string.
        
        Args:
            fig: Matplotlib figure object
            
        Returns:
            Base64 encoded string of the figure
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return img_str