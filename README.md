# SmartSynth: Domain-Agnostic Synthetic Data Generator

SmartSynth is a comprehensive, scalable synthetic data generation framework designed to create high-quality, privacy-preserving synthetic data across multiple domains and data types. This project provides both a command-line interface and a web-based UI for generating, evaluating, and visualizing synthetic data.

## Features

- **Multi-domain support**: Generate synthetic data for tabular, time-series, text, and image data
- **Multiple generation techniques**: Includes CTGAN, TVAE, CopulaGAN, TimeGAN, and transformer-based text generation
- **Privacy preservation**: Built-in differential privacy, k-anonymity, and membership inference protection
- **Comprehensive evaluation**: Statistical similarity, machine learning utility, and privacy risk assessment
- **Interactive visualization**: Compare original and synthetic data distributions and properties
- **User-friendly interfaces**: Both command-line and web-based UI options

## Project Structure

```
syn_data/
├── app.py                  # Streamlit web application
├── cli.py                  # Command-line interface
├── models/                 # Generator implementations
│   ├── __init__.py         # Package initialization
│   ├── base_generator.py   # Abstract base generator class
│   ├── generator_factory.py # Factory for creating generators
│   ├── ctgan_generator.py  # CTGAN implementation
│   ├── tvae_generator.py   # TVAE implementation
│   ├── copulagan_generator.py # CopulaGAN implementation
│   ├── timegan_generator.py # TimeGAN implementation
│   ├── text_generator.py   # Text generator implementation
│   ├── image_generator.py  # Image generator implementation
│   └── privacy_filter.py   # Privacy preservation techniques
├── utils/                  # Utility modules
│   ├── __init__.py         # Package initialization
│   ├── profiler.py         # Data profiling utilities
│   ├── visualizer.py       # Visualization utilities
│   └── evaluator.py        # Evaluation metrics and reporting
└── README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vi14m/syn_data.git
cd syn_data

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface

```bash
streamlit run app.py
```

This will launch the Streamlit web application, which provides an intuitive interface for:
1. Uploading and profiling data
2. Configuring and generating synthetic data
3. Evaluating and comparing original vs. synthetic data
4. Downloading the generated synthetic data

### Command Line Interface

```bash
# Basic usage
python cli.py --data path/to/data.csv --output path/to/output.csv --generator ctgan --samples 1000

# With privacy filtering
python cli.py --data path/to/data.csv --output path/to/output.csv --generator ctgan --samples 1000 --privacy differential_privacy

# Save and load models
python cli.py --data path/to/data.csv --save-model path/to/model.pkl --generator ctgan
python cli.py --load-model path/to/model.pkl --generator ctgan --samples 1000 --output path/to/output.csv

# Generate profile report
python cli.py --data path/to/data.csv --profile

# Generate visualizations
python cli.py --data path/to/data.csv --generator ctgan --samples 1000 --visualize path/to/viz/folder
```

## Supported Generator Types

- **CTGAN**: Conditional Tabular GAN for tabular data
- **TVAE**: Tabular Variational Autoencoder for tabular data
- **CopulaGAN**: Copula-based GAN for preserving statistical relationships
- **TimeGAN**: Time-series GAN for sequential data
- **Text Generator**: Transformer-based text generation (GPT-2 by default)
- **Image Generator**: Diffusion model-based image generation (Stable Diffusion by default)

## Privacy Preservation Methods

- **Differential Privacy**: Adds calibrated noise to ensure individual privacy
- **k-anonymity**: Ensures each record is indistinguishable from at least k-1 other records
- **Membership Inference Protection**: Prevents attackers from determining if a record was in the training data

## Evaluation Metrics

- **Statistical Similarity**: KS test, Wasserstein distance, Jensen-Shannon divergence
- **Machine Learning Utility**: Model performance comparison, feature importance similarity
- **Privacy Risk**: Distance-based metrics, membership inference risk, attribute disclosure risk
- **Overall Quality Score**: Weighted combination with letter grade (A+ to F)

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- sdv (Synthetic Data Vault)
- ydata-synthetic
- transformers
- diffusers
- torch
- streamlit

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.