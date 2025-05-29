import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Any

# Import base generator class
from .base_generator import BaseGenerator

class TextGenerator(BaseGenerator):
    """Text generator for synthetic text data using transformer models."""
    
    def __init__(self, **kwargs):
        """Initialize the Text generator.
        
        Args:
            **kwargs: Additional arguments to pass to the text generator constructor
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        self.model_name = kwargs.get('model_name', 'gpt2')
        self.model_kwargs = kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def configure(self, max_length: int = 100, temperature: float = 0.7, 
                 top_p: float = 0.9, top_k: int = 50, **kwargs):
        """Configure the text generator.
        
        Args:
            max_length: Maximum length of generated text
            temperature: Temperature for sampling (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional configuration parameters
        """
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured Text Generator with max_length={max_length}, "
                        f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
    
    def fit(self, data):
        """Fit the text generator model to the input data.
        
        Args:
            data: Input text data to learn from. Can be a list of strings, a Series, or a DataFrame with a text column.
            
        Returns:
            self: The fitted generator
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
            from datasets import Dataset
            
            # Process input data to get a list of texts
            texts = self._process_input_data(data)
            self.logger.info(f"Processed {len(texts)} text samples for training")
            
            # Load pre-trained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            # For fine-tuning (optional, based on data size)
            if len(texts) > 100 and self.model_kwargs.get('fine_tune', True):
                self.logger.info(f"Fine-tuning {self.model_name} on {len(texts)} text samples")
                
                # Tokenize the texts
                def tokenize_function(examples):
                    return self.tokenizer(examples["text"], padding="max_length", truncation=True)
                
                # Create a dataset
                dataset = Dataset.from_dict({"text": texts})
                tokenized_dataset = dataset.map(tokenize_function, batched=True)
                
                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir="./results",
                    num_train_epochs=3,
                    per_device_train_batch_size=4,
                    save_steps=500,
                    save_total_limit=2,
                )
                
                # Create Trainer and train
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                )
                
                trainer.train()
                self.logger.info("Fine-tuning completed")
            else:
                self.logger.info(f"Using pre-trained {self.model_name} without fine-tuning")
            
            return self
            
        except ImportError:
            self.logger.error("Required packages not found. Please install them with 'pip install transformers datasets torch'")
            raise
        except Exception as e:
            self.logger.error(f"Error fitting text generator model: {str(e)}")
            raise
    
    def generate(self, num_samples: int, prompts: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate synthetic text samples using the fitted model.
        
        Args:
            num_samples: Number of synthetic text samples to generate
            prompts: Optional list of prompts to start generation from
            
        Returns:
            DataFrame containing the generated synthetic text samples
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            self.logger.info(f"Generating {num_samples} synthetic text samples")
            
            # Use provided prompts or generate from empty strings
            if prompts is None:
                prompts = ["" for _ in range(num_samples)]
            elif len(prompts) < num_samples:
                # Repeat prompts if needed
                prompts = (prompts * ((num_samples // len(prompts)) + 1))[:num_samples]
            
            generated_texts = []
            
            for prompt in prompts:
                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                
                # Generate text
                output = self.model.generate(
                    input_ids,
                    max_length=self.max_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                generated_texts.append(generated_text)
            
            # Create DataFrame
            synthetic_df = pd.DataFrame({"generated_text": generated_texts})
            
            self.logger.info(f"Generated {len(synthetic_df)} synthetic text samples")
            
            return synthetic_df
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic text: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save the fitted text generator model to disk.
        
        Args:
            path: Path to save the model to
            
        Raises:
            RuntimeError: If the model has not been fitted yet
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save configuration
            config = {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "model_kwargs": self.model_kwargs
            }
            
            import json
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config, f)
            
            self.logger.info(f"Saved text generator model to {path}")
        except Exception as e:
            self.logger.error(f"Error saving text generator model: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load a fitted text generator model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Load configuration
            import json
            with open(os.path.join(path, "config.json"), "r") as f:
                config = json.load(f)
            
            self.model_name = config["model_name"]
            self.max_length = config["max_length"]
            self.temperature = config["temperature"]
            self.top_p = config["top_p"]
            self.top_k = config["top_k"]
            self.model_kwargs = config["model_kwargs"]
            
            self.logger.info(f"Loaded text generator model from {path}")
        except Exception as e:
            self.logger.error(f"Error loading text generator model: {str(e)}")
            raise
    
    def _process_input_data(self, data):
        """Process input data to get a list of texts.
        
        Args:
            data: Input text data (list, Series, or DataFrame)
            
        Returns:
            List of text strings
        """
        if isinstance(data, list):
            # Already a list of strings
            return data
        elif isinstance(data, pd.Series):
            # Convert Series to list
            return data.tolist()
        elif isinstance(data, pd.DataFrame):
            # Try to find a text column
            text_columns = [col for col in data.columns if any(term in col.lower() for term in 
                                                           ['text', 'content', 'message', 'description'])]
            
            if text_columns:
                # Use the first text column
                return data[text_columns[0]].tolist()
            else:
                # Use the first column
                self.logger.warning("No obvious text column found, using the first column")
                return data.iloc[:, 0].tolist()
        else:
            raise ValueError("Unsupported data type. Please provide a list of strings, a Series, or a DataFrame with a text column.")