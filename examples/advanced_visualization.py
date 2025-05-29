import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.subplots as sp
from pandas_profiling import ProfileReport
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata.single_table import SingleTableMetadata

def create_profiling_report(real_data, title="Profiling Real Data", output_file="real_data_report.html"):
    """
    Create a detailed profiling report using pandas-profiling.
    
    Args:
        real_data: DataFrame containing the real data
        title: Title for the profiling report
        output_file: Path to save the HTML report
        
    Returns:
        ProfileReport object
    """
    profile = ProfileReport(
        real_data, 
        title=title, 
        html={"style": {"full_width": True}}
    )
    
    # Save the report to file
    profile.to_file(output_file)
    
    return profile

def create_column_comparison_plots(real_data, synthetic_data, metadata, columns=None):
    """
    Create comparison plots between real and synthetic data for each column.
    
    Args:
        real_data: DataFrame containing the real data
        synthetic_data: DataFrame containing the synthetic data
        metadata: SingleTableMetadata object with data types information
        columns: List of columns to visualize (if None, all columns are used)
        
    Returns:
        List of plotly figures for each column
    """
    if columns is None:
        columns = real_data.columns
    
    # Generate column plots
    figs = []
    for column in columns:
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name=column,
            metadata=metadata
        )
        figs.append(fig)
    
    return figs

def create_subplot_grid(figures, rows=2, cols=3, height=600, width=900, title='Column Comparison'):
    """
    Arrange multiple column plots in a grid of subplots.
    
    Args:
        figures: List of plotly figures to arrange
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        height: Height of the figure in pixels
        width: Width of the figure in pixels
        title: Title for the overall figure
        
    Returns:
        Combined plotly figure with subplots
    """
    # Create subplot titles using the first figure's title from each column plot
    subplot_titles = [fig.layout.title.text for fig in figures[:rows*cols]]
    
    # Create subplot grid
    fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
    
    # Add traces from individual figures to the subplot grid
    for i in range(min(rows * cols, len(figures))):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Add both real and synthetic data traces
        for trace in figures[i]['data']:
            fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=height, 
        width=width, 
        title_text=title
    )
    
    return fig

def visualize_data_comparison(real_data, synthetic_data, metadata=None, output_dir="visualizations"):
    """
    Generate comprehensive visualizations comparing real and synthetic data.
    
    Args:
        real_data: DataFrame containing the real data
        synthetic_data: DataFrame containing the synthetic data
        metadata: SingleTableMetadata object (if None, it will be created)
        output_dir: Directory to save visualization outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata if not provided
    if metadata is None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
    
    # Generate profiling report
    profile_path = os.path.join(output_dir, "real_data_report.html")
    profile = create_profiling_report(
        real_data, 
        title="Profiling Real Data", 
        output_file=profile_path
    )
    print(f"Profiling report saved to {profile_path}")
    
    # Generate column comparison plots
    column_figs = create_column_comparison_plots(real_data, synthetic_data, metadata)
    
    # Create subplot grids (multiple if needed)
    max_cols_per_grid = 6
    for i in range(0, len(column_figs), max_cols_per_grid):
        batch_figs = column_figs[i:i+max_cols_per_grid]
        rows = 2
        cols = 3
        
        # Adjust grid dimensions for smaller batches
        if len(batch_figs) <= 3:
            rows = 1
            cols = len(batch_figs)
        
        # Create and show subplot grid
        grid_fig = create_subplot_grid(
            batch_figs, 
            rows=rows, 
            cols=cols, 
            title=f'Column Comparison (Batch {i//max_cols_per_grid + 1})'
        )
        
        # Save the figure
        grid_path = os.path.join(output_dir, f"column_comparison_batch_{i//max_cols_per_grid + 1}.html")
        grid_fig.write_html(grid_path)
        print(f"Column comparison batch {i//max_cols_per_grid + 1} saved to {grid_path}")
        
        # Display the figure
        grid_fig.show()

# Example usage
if __name__ == "__main__":
    # Load your data
    # real_data = pd.read_csv("path/to/real_data.csv")
    # synthetic_data = pd.read_csv("path/to/synthetic_data.csv")
    
    # Create metadata
    # metadata = SingleTableMetadata()
    # metadata.detect_from_dataframe(real_data)
    
    # Run visualization
    # visualize_data_comparison(real_data, synthetic_data, metadata)
    
    print("Import this module and use the functions to create advanced visualizations")