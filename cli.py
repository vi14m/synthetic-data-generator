#!/usr/bin/env python

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional

# Import project modules
from models import (
    GeneratorFactory, BaseGenerator, CTGANGenerator, TVAEGenerator, 
    CopulaGANGenerator, TimeGANGenerator, TextGenerator, ImageGenerator,
    PrivacyFilter
)
from utils import DataProfiler, DataVisualizer, DataEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('synthetic_data_generator.log')
    ]
)

logger = logging.getLogger(__name__)

class SyntheticDataGeneratorCLI:
    """Command-line interface for the synthetic data generator."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        self.args = None
        self.generator = None
        self.original_data = None
        self.synthetic_data = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser.
        
        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(description="SmartSynth: Domain-Agnostic Synthetic Data Generator")
        
        # Data input/output arguments
        parser.add_argument("--data", type=str, help="Path to input data file")
        parser.add_argument("--output", type=str, help="Path to save synthetic data")
        parser.add_argument("--format", type=str, default="csv", 
                            choices=["csv", "parquet", "excel", "json", "pickle"],
                            help="Output file format")
        
        # Generator configuration
        parser.add_argument("--generator", type=str, default="ctgan", 
                            choices=["ctgan", "tvae", "copulagan", "timegan", "text", "image"],
                            help="Generator type to use")
        parser.add_argument("--samples", type=int, default=1000, 
                            help="Number of synthetic samples to generate")
        parser.add_argument("--config", type=str, help="Path to configuration file")
        
        # Model saving/loading
        parser.add_argument("--save-model", type=str, help="Path to save trained model")
        parser.add_argument("--load-model", type=str, help="Path to load trained model")
        
        # Analysis and visualization
        parser.add_argument("--profile", action="store_true", 
                            help="Generate data profile report")
        parser.add_argument("--profile-output", type=str, 
                            help="Path to save profile report")
        parser.add_argument("--visualize", type=str, 
                            help="Path to save visualization files")
        parser.add_argument("--evaluate", action="store_true", 
                            help="Evaluate synthetic data quality")
        parser.add_argument("--evaluation-output", type=str, 
                            help="Path to save evaluation report")
        
        # Privacy options
        parser.add_argument("--privacy", type=str, 
                            choices=["differential_privacy", "k_anonymity", "membership_inference_protection"],
                            help="Apply privacy filtering method")
        parser.add_argument("--epsilon", type=float, default=1.0, 
                            help="Epsilon value for differential privacy")
        parser.add_argument("--k", type=int, default=5, 
                            help="k value for k-anonymity")
        
        # Advanced generator options
        parser.add_argument("--epochs", type=int, default=300, 
                            help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=500, 
                            help="Batch size for training")
        parser.add_argument("--embedding-dim", type=int, default=128, 
                            help="Embedding dimension for generator")
        
        return parser
    
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse command-line arguments.
        
        Args:
            args: Command-line arguments (defaults to sys.argv)
            
        Returns:
            Parsed arguments
        """
        self.args = self.parser.parse_args(args)
        return self.args
    
    def run(self) -> int:
        """Run the CLI application.
        
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        if self.args is None:
            self.parse_args()
        
        try:
            # Load data if provided
            if self.args.data:
                self._load_data()
                
                # Generate profile report if requested
                if self.args.profile:
                    self._profile_data()
            
            # Load or train model
            if self.args.load_model:
                self._load_model()
            elif self.original_data is not None:
                self._initialize_and_train_generator()
                
                # Save model if requested
                if self.args.save_model:
                    self._save_model()
            
            # Generate synthetic data
            if self.generator is not None:
                self._generate_synthetic_data()
                
                # Apply privacy filtering if requested
                if self.args.privacy:
                    self._apply_privacy_filter()
                
                # Evaluate synthetic data if requested
                if self.args.evaluate:
                    self._evaluate_synthetic_data()
                
                # Generate visualizations if requested
                if self.args.visualize:
                    self._visualize_data_comparison()
                
                # Save synthetic data if output path provided
                if self.args.output:
                    self._save_synthetic_data()
            
            return 0  # Success
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return 1  # Error
    
    def _load_data(self) -> None:
        """Load data from file."""
        logger.info(f"Loading data from {self.args.data}")
        
        file_extension = os.path.splitext(self.args.data)[1].lower()
        
        if file_extension == '.csv':
            import pandas as pd
            self.original_data = pd.read_csv(self.args.data)
        elif file_extension == '.parquet':
            import pandas as pd
            self.original_data = pd.read_parquet(self.args.data)
        elif file_extension in ['.xls', '.xlsx']:
            import pandas as pd
            self.original_data = pd.read_excel(self.args.data)
        elif file_extension == '.json':
            import pandas as pd
            self.original_data = pd.read_json(self.args.data)
        elif file_extension == '.pkl':
            import pandas as pd
            self.original_data = pd.read_pickle(self.args.data)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        logger.info(f"Loaded data with shape {self.original_data.shape}")
    
    def _profile_data(self) -> None:
        """Profile the loaded data."""
        logger.info("Profiling data")
        
        profiler = DataProfiler(self.original_data)
        profile = profiler.profile_data()
        
        # Print basic statistics
        print("\nData Profile:")
        print(f"Rows: {profile['row_count']}")
        print(f"Columns: {profile['column_count']}")
        print(f"Missing values: {profile['missing_percentage']:.2f}%")
        
        # Save profile report if requested
        if self.args.profile_output:
            profiler.save_profile_report(self.args.profile_output)
            logger.info(f"Saved profile report to {self.args.profile_output}")
        else:
            # Default path
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"data_profile_{timestamp}.html"
            profiler.save_profile_report(report_path)
            logger.info(f"Saved profile report to {report_path}")
    
    def _initialize_and_train_generator(self) -> None:
        """Initialize and train the generator."""
        logger.info(f"Initializing {self.args.generator} generator")
        
        # Create generator using factory
        factory = GeneratorFactory()
        self.generator = factory.get_generator(
            self.args.generator,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            embedding_dim=self.args.embedding_dim
        )
        
        logger.info("Training generator")
        self.generator.fit(self.original_data)
        logger.info("Generator training completed")
    
    def _load_model(self) -> None:
        """Load a trained generator model."""
        logger.info(f"Loading {self.args.generator} model from {self.args.load_model}")
        
        # Create generator using factory
        factory = GeneratorFactory()
        self.generator = factory.create_generator(self.args.generator)
        
        # Load the model
        self.generator.load(self.args.load_model)
        logger.info("Model loaded successfully")
    
    def _save_model(self) -> None:
        """Save the trained generator model."""
        logger.info(f"Saving model to {self.args.save_model}")
        self.generator.save(self.args.save_model)
        logger.info("Model saved successfully")
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data using the trained generator."""
        logger.info(f"Generating {self.args.samples} synthetic samples")
        self.synthetic_data = self.generator.generate(self.args.samples)
        logger.info(f"Generated synthetic data with shape {self.synthetic_data.shape}")
    
    def _apply_privacy_filter(self) -> None:
        """Apply privacy filtering to the synthetic data."""
        logger.info(f"Applying {self.args.privacy} privacy filtering")
        
        privacy_filter = PrivacyFilter()
        
        if self.args.privacy == "differential_privacy":
            self.synthetic_data = privacy_filter.apply_differential_privacy(
                self.synthetic_data, epsilon=self.args.epsilon
            )
        elif self.args.privacy == "k_anonymity":
            self.synthetic_data = privacy_filter.apply_k_anonymity(
                self.synthetic_data, k=self.args.k
            )
        elif self.args.privacy == "membership_inference_protection":
            self.synthetic_data = privacy_filter.protect_against_membership_inference(
                self.original_data, self.synthetic_data
            )
        
        logger.info("Privacy filtering applied successfully")
    
    def _evaluate_synthetic_data(self) -> None:
        """Evaluate the quality of the synthetic data."""
        logger.info("Evaluating synthetic data quality")
        
        evaluator = DataEvaluator(self.original_data, self.synthetic_data)
        evaluation = evaluator.get_overall_quality_score()
        
        # Print evaluation results
        print("\nSynthetic Data Evaluation:")
        print(f"Overall Quality Score: {evaluation['overall']:.2f} ({evaluation['grade']})")
        print(f"Statistical Similarity: {evaluation['statistical_similarity']:.2f}")
        if 'ml_utility' in evaluation:
            print(f"ML Utility: {evaluation['ml_utility']:.2f}")
        if 'privacy' in evaluation:
            print(f"Privacy Score: {evaluation['privacy']:.2f}")
        
        # Save evaluation report if requested
        if self.args.evaluation_output:
            import json
            with open(self.args.evaluation_output, 'w') as f:
                json.dump(evaluation, f, indent=4)
            logger.info(f"Saved evaluation report to {self.args.evaluation_output}")
    
    def _visualize_data_comparison(self) -> None:
        """Visualize comparison between original and synthetic data."""
        logger.info("Generating data comparison visualizations")
        
        visualizer = DataVisualizer(self.original_data, self.synthetic_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.args.visualize, exist_ok=True)
        
        # Generate and save visualizations
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Distribution plots
        dist_plots = visualizer.plot_distributions()
        for i, (col, plot) in enumerate(dist_plots.items()):
            plot_path = os.path.join(self.args.visualize, f"dist_{i}_{col}_{timestamp}.png")
            plot.savefig(plot_path)
        
        # Correlation heatmaps
        corr_plots = visualizer.plot_correlation_heatmaps()
        for i, (name, plot) in enumerate(corr_plots.items()):
            plot_path = os.path.join(self.args.visualize, f"corr_{name}_{timestamp}.png")
            plot.savefig(plot_path)
        
        logger.info(f"Saved visualizations to {self.args.visualize}")
    
    def _save_synthetic_data(self) -> None:
        """Save the generated synthetic data to file."""
        logger.info(f"Saving synthetic data to {self.args.output} in {self.args.format} format")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        
        # Save data in the specified format
        format = self.args.format.lower()
        if format == "csv":
            self.synthetic_data.to_csv(self.args.output, index=False)
        elif format == "parquet":
            self.synthetic_data.to_parquet(self.args.output, index=False)
        elif format == "excel":
            self.synthetic_data.to_excel(self.args.output, index=False)
        elif format == "json":
            self.synthetic_data.to_json(self.args.output, orient="records")
        elif format == "pickle":
            self.synthetic_data.to_pickle(self.args.output)
        
        logger.info("Synthetic data saved successfully")


if __name__ == "__main__":
    cli = SyntheticDataGeneratorCLI()
    sys.exit(cli.run())