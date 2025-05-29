import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Import custom modules
from utils.profiler import DataProfiler
from utils.visualizer import DataVisualizer
from utils.evaluator import DataEvaluator
from models.generator_factory import GeneratorFactory
from models.privacy_filter import PrivacyFilter

# Set page configuration
st.set_page_config(
    page_title="SmartSynth - Synthetic Data Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'data_profile' not in st.session_state:
    st.session_state.data_profile = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'generator_type' not in st.session_state:
    st.session_state.generator_type = None

# App title and description
st.title("SmartSynth: Domain-Agnostic Synthetic Data Generator")
st.markdown("""
    Generate high-quality, privacy-preserving synthetic data for various domains and modalities.
    Upload your dataset, configure generation parameters, and download realistic synthetic data.
""")

# Sidebar for navigation and configuration
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Profile", "Configure & Generate", "Evaluate & Compare", "Download"])

# Upload & Profile page
if page == "Upload & Profile":
    st.header("Upload & Profile Your Data")
    
    # File uploader
    data_type = st.radio("Select data type", ["Tabular (CSV/Excel)", "Time-Series", "Text", "Image"], horizontal=True)
    
    if data_type == "Tabular (CSV/Excel)":
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = data
                st.session_state.generator_type = "tabular"
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Data profiling
                st.subheader("Data Profiling")
                with st.spinner("Profiling data... This may take a moment."):
                    profiler = DataProfiler(data)
                    profile = profiler.generate_profile()
                    st.session_state.data_profile = profile
                    
                    # Display basic statistics
                    st.write(f"**Rows:** {profile['row_count']}")
                    st.write(f"**Columns:** {profile['column_count']}")
                    st.write(f"**Missing values:** {profile['missing_percentage']:.2f}%")
                    
                    # Display column types
                    st.write("**Column Types:**")
                    col_types = pd.DataFrame(profile['column_types'].items(), columns=['Column', 'Type'])
                    st.dataframe(col_types)
                    
                    # Display correlations
                    st.write("**Correlation Heatmap:**")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(profile['correlation_matrix'], annot=False, cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif data_type == "Time-Series":
        st.info("Time-Series data support is coming soon! Please use tabular data for now.")
        
    elif data_type == "Text":
        st.info("Text data support is coming soon! Please use tabular data for now.")
        
    elif data_type == "Image":
        st.info("Image data support is coming soon! Please use tabular data for now.")

# Configure & Generate page
elif page == "Configure & Generate":
    st.header("Configure & Generate Synthetic Data")
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload data first in the 'Upload & Profile' page.")
    else:
        st.subheader("Generation Parameters")
        
        if st.session_state.generator_type == "tabular":
            # Generator selection
            generator_model = st.selectbox(
                "Select generator model", 
                ["CTGAN (Conditional Tabular GAN)", "TVAE (Tabular Variational Autoencoder)", "CopulaGAN"]
            )
            
            # Number of synthetic samples
            num_samples = st.number_input(
                "Number of synthetic samples to generate", 
                min_value=1, 
                max_value=100000, 
                value=min(len(st.session_state.uploaded_data) * 2, 10000)
            )
            
            # Privacy settings
            st.subheader("Privacy Settings")
            enable_dp = st.checkbox("Enable Differential Privacy")
            
            if enable_dp:
                epsilon = st.slider(
                    "Privacy Budget (ε) - Lower values provide stronger privacy", 
                    min_value=0.1, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1
                )
                delta = st.number_input(
                    "Delta (δ) - Probability of privacy breach", 
                    min_value=0.00001, 
                    max_value=0.1, 
                    value=0.001, 
                    format="%f"
                )
            else:
                epsilon = None
                delta = None
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                epochs = st.slider("Training Epochs", min_value=10, max_value=500, value=300)
                batch_size = st.slider("Batch Size", min_value=100, max_value=1000, value=500, step=100)
                embedding_dim = st.slider("Embedding Dimension", min_value=32, max_value=256, value=128, step=32)
            
            # Generate button
            if st.button("Generate Synthetic Data"):
                with st.spinner("Generating synthetic data... This may take a few minutes."):
                    try:
                        # Create generator factory
                        factory = GeneratorFactory()
                        
                        # Get appropriate generator
                        if generator_model == "CTGAN (Conditional Tabular GAN)":
                            generator = factory.get_generator("ctgan")
                        elif generator_model == "TVAE (Tabular Variational Autoencoder)":
                            generator = factory.get_generator("tvae")
                        else:  # CopulaGAN
                            generator = factory.get_generator("copulagan")
                        
                        # Configure generator
                        generator.configure(
                            epochs=epochs,
                            batch_size=batch_size,
                            embedding_dim=embedding_dim
                        )
                        
                        # Fit generator to data
                        generator.fit(st.session_state.uploaded_data)
                        
                        # Generate synthetic data
                        synthetic_data = generator.generate(num_samples)
                        
                        # Apply privacy filter if enabled
                        if enable_dp:
                            privacy_filter = PrivacyFilter(epsilon=epsilon, delta=delta)
                            synthetic_data = privacy_filter.apply(synthetic_data, st.session_state.uploaded_data)
                        
                        # Store synthetic data in session state
                        st.session_state.synthetic_data = synthetic_data
                        
                        # Display success message
                        st.success(f"Successfully generated {len(synthetic_data)} synthetic samples!")
                        
                        # Show preview
                        st.subheader("Synthetic Data Preview")
                        st.dataframe(synthetic_data.head())
                        
                    except Exception as e:
                        st.error(f"Error generating synthetic data: {e}")
        else:
            st.info(f"Support for {st.session_state.generator_type} data is coming soon!")

# Evaluate & Compare page
elif page == "Evaluate & Compare":
    st.header("Evaluate & Compare Synthetic Data")
    
    if st.session_state.synthetic_data is None:
        st.warning("Please generate synthetic data first in the 'Configure & Generate' page.")
    else:
        # Initialize evaluator
        evaluator = DataEvaluator(
            original_data=st.session_state.uploaded_data,
            synthetic_data=st.session_state.synthetic_data
        )
        
        # Run evaluation
        with st.spinner("Evaluating synthetic data quality... This may take a moment."):
            evaluation_results = evaluator.get_overall_quality_score()
            st.session_state.evaluation_results = evaluation_results
        
        # Display evaluation results
        st.subheader("Statistical Similarity")
        
        # Visualization options
        viz_type = st.radio(
            "Select visualization type",
            ["Basic Comparison", "Advanced Visualization"],
            horizontal=True
        )
        
        if viz_type == "Basic Comparison":
            # Column distribution comparison
            st.write("**Column Distribution Comparison:**")
            selected_column = st.selectbox(
                "Select column to compare", 
                st.session_state.uploaded_data.columns
            )
            
            # Create visualization
            visualizer = DataVisualizer(
                original_data=st.session_state.uploaded_data,
                synthetic_data=st.session_state.synthetic_data
            )
            
            fig = visualizer.get_distribution_plot(selected_column)
            st.pyplot(fig)
        else:
            # Import advanced visualization module
            from utils.advanced_visualization import display_advanced_visualizations_streamlit
            
            # Display advanced visualizations
            with st.spinner("Generating advanced visualizations... This may take a moment."):
                display_advanced_visualizations_streamlit(
                    real_data=st.session_state.uploaded_data,
                    synthetic_data=st.session_state.synthetic_data
                )
        
        # Evaluation metrics
        st.subheader("Evaluation Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Statistical Similarity', 'ML Utility', 'Privacy', 'Overall Score', 'Grade'],
            'Value': [evaluation_results['statistical_similarity'], 
                     evaluation_results['ml_utility'], 
                     evaluation_results['privacy'], 
                     evaluation_results['overall'], 
                     evaluation_results['grade']]
        })
        st.dataframe(metrics_df)
        
        # Privacy evaluation
        st.subheader("Privacy Evaluation")
        st.write(f"**Privacy Score:** {evaluation_results['privacy']:.4f}")
        
        # Interpretation
        st.info("""
        **Interpretation Guide:**
        - **Statistical Similarity:** Higher values (closer to 1.0) indicate better similarity between real and synthetic data distributions.
        - **Nearest Neighbor Distance:** Higher values indicate better privacy (synthetic data points are further from real data points).
        - **Membership Inference AUC:** Values closer to 0.5 indicate better privacy (harder to determine if a record was in the training data).
        """)

# Download page
elif page == "Download":
    st.header("Download Synthetic Data")
    
    if st.session_state.synthetic_data is None:
        st.warning("Please generate synthetic data first in the 'Configure & Generate' page.")
    else:
        st.subheader("Export Options")
        
        # File format selection
        file_format = st.radio("Select file format", ["CSV", "JSON", "Excel"], horizontal=True)
        
        # Prepare download button
        if file_format == "CSV":
            csv = st.session_state.synthetic_data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif file_format == "JSON":
            json_str = st.session_state.synthetic_data.to_json(orient="records")
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="synthetic_data.json">Download JSON File</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif file_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.synthetic_data.to_excel(writer, index=False, sheet_name='Synthetic Data')
            b64 = base64.b64encode(output.getvalue()).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="synthetic_data.xlsx">Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Additional information
        with st.expander("Synthetic Data Summary"):
            st.write(f"**Number of records:** {len(st.session_state.synthetic_data)}")
            st.write(f"**Number of columns:** {len(st.session_state.synthetic_data.columns)}")
            st.write("**Column types:**")
            st.dataframe(st.session_state.synthetic_data.dtypes.reset_index().rename(
                columns={"index": "Column", 0: "Data Type"}
            ))

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>SmartSynth: A Domain-Agnostic Synthetic Data Generator | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)