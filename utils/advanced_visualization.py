import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.subplots as sp
import streamlit as st
from ydata_profiling import ProfileReport
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata.single_table import SingleTableMetadata

def create_profiling_report(real_data, title="Profiling Real Data", output_file="real_data_report.html"):
    """
    Create a detailed profiling report using ydata-profiling (formerly pandas-profiling).
    
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
    
    Returns:
        Dictionary containing paths to generated visualizations
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
    
    # Generate column comparison plots
    column_figs = create_column_comparison_plots(real_data, synthetic_data, metadata)
    
    # Create subplot grids (multiple if needed)
    grid_paths = []
    max_cols_per_grid = 6
    for i in range(0, len(column_figs), max_cols_per_grid):
        batch_figs = column_figs[i:i+max_cols_per_grid]
        rows = 2
        cols = 3
        
        # Adjust grid dimensions for smaller batches
        if len(batch_figs) <= 3:
            rows = 1
            cols = len(batch_figs)
        
        # Create subplot grid
        grid_fig = create_subplot_grid(
            batch_figs, 
            rows=rows, 
            cols=cols, 
            title=f'Column Comparison (Batch {i//max_cols_per_grid + 1})'
        )
        
        # Save the figure
        grid_path = os.path.join(output_dir, f"column_comparison_batch_{i//max_cols_per_grid + 1}.html")
        grid_fig.write_html(grid_path)
        grid_paths.append(grid_path)
    
    return {
        "profile_report": profile_path,
        "column_comparisons": grid_paths,
        "column_figures": column_figs
    }

def display_advanced_visualizations_streamlit(real_data, synthetic_data, metadata=None):
    """
    Display advanced visualizations in Streamlit app.
    
    Args:
        real_data: DataFrame containing the real data
        synthetic_data: DataFrame containing the synthetic data
        metadata: SingleTableMetadata object (if None, it will be created)
    """
    # Create metadata if not provided
    if metadata is None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(real_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Column Comparisons", "Real Data Profile", "Synthetic Data Profile"])
    
    with tab1:
        st.subheader("Column Distribution Comparisons")
        
        # Generate column comparison plots
        column_figs = create_column_comparison_plots(real_data, synthetic_data, metadata)
        
        # Allow user to select columns to display
        selected_columns = st.multiselect(
            "Select columns to compare",
            options=real_data.columns,
            default=list(real_data.columns[:min(6, len(real_data.columns))])
        )
        
        if selected_columns:
            # Filter figures based on selected columns
            selected_figs = [fig for fig, col in zip(column_figs, real_data.columns) if col in selected_columns]
            
            # Determine grid dimensions
            num_figs = len(selected_figs)
            if num_figs <= 3:
                rows, cols = 1, num_figs
            else:
                rows, cols = (num_figs + 2) // 3, 3  # Ceiling division to get number of rows
            
            # Create subplot grid
            grid_fig = create_subplot_grid(
                selected_figs,
                rows=rows,
                cols=cols,
                height=rows*300,  # Adjust height based on number of rows
                width=900,
                title='Column Distribution Comparison'
            )
            
            # Display the figure
            st.plotly_chart(grid_fig, use_container_width=True)
        else:
            st.info("Please select at least one column to display comparison plots.")
    
    with tab2:
        st.subheader("Real Data Profile Report")
        st.write("Generating comprehensive profile report...")
        
        # Create a temporary directory for the profile report
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "real_profile_report.html")
            profile = create_profiling_report(real_data, title="Real Data Profile", output_file=profile_path)
            
            # Display the HTML report in an iframe
            with open(profile_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
    
    with tab3:
        st.subheader("Synthetic Data Profile Report")
        st.write("Generating comprehensive profile report...")
        
        # Create a temporary directory for the profile report
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "synthetic_profile_report.html")
            profile = create_profiling_report(synthetic_data, title="Synthetic Data Profile", output_file=profile_path)
            
            # Display the HTML report in an iframe
            with open(profile_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)