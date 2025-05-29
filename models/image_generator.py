import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Any
from PIL import Image

# Import base generator class
from .base_generator import BaseGenerator

class ImageGenerator(BaseGenerator):
    """Image generator for synthetic image data using diffusion models."""
    
    def __init__(self, **kwargs):
        """Initialize the Image generator.
        
        Args:
            **kwargs: Additional arguments to pass to the image generator constructor
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.pipeline = None
        self.model_name = kwargs.get('model_name', 'stabilityai/stable-diffusion-2-1')
        self.model_kwargs = kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def configure(self, image_size: int = 512, guidance_scale: float = 7.5, 
                 num_inference_steps: int = 50, **kwargs):
        """Configure the image generator.
        
        Args:
            image_size: Size of generated images (square)
            guidance_scale: Guidance scale for diffusion model
            num_inference_steps: Number of inference steps
            **kwargs: Additional configuration parameters
        """
        self.image_size = image_size
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.model_kwargs.update(kwargs)
        self.logger.info(f"Configured Image Generator with image_size={image_size}, "
                        f"guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}")
    
    def fit(self, data=None):
        """Load the pre-trained diffusion model.
        
        Args:
            data: Optional input image data for fine-tuning (not implemented yet)
            
        Returns:
            self: The loaded generator
        """
        try:
            from diffusers import StableDiffusionPipeline, DiffusionPipeline
            
            self.logger.info(f"Loading pre-trained diffusion model: {self.model_name}")
            
            # Load pre-trained diffusion model
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
            
            # Enable attention slicing for lower memory usage
            if self.device.type == 'cuda':
                self.pipeline.enable_attention_slicing()
            
            self.logger.info(f"Successfully loaded diffusion model: {self.model_name}")
            
            # Fine-tuning is not implemented yet, but could be added in the future
            if data is not None:
                self.logger.warning("Fine-tuning for image generation is not implemented yet. Using pre-trained model.")
            
            return self
            
        except ImportError:
            self.logger.error("Required packages not found. Please install them with 'pip install diffusers transformers torch'")
            raise
        except Exception as e:
            self.logger.error(f"Error loading diffusion model: {str(e)}")
            raise
    
    def generate(self, num_samples: int, prompts: Optional[List[str]] = None) -> List[Image.Image]:
        """Generate synthetic image samples using the diffusion model.
        
        Args:
            num_samples: Number of synthetic image samples to generate
            prompts: Optional list of text prompts to guide generation
            
        Returns:
            List of PIL Image objects
            
        Raises:
            RuntimeError: If the model has not been loaded yet
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been loaded yet. Call fit() first.")
        
        try:
            self.logger.info(f"Generating {num_samples} synthetic image samples")
            
            # Use provided prompts or generate from default prompt
            if prompts is None:
                prompts = ["a high-quality, detailed synthetic image" for _ in range(num_samples)]
            elif len(prompts) < num_samples:
                # Repeat prompts if needed
                prompts = (prompts * ((num_samples // len(prompts)) + 1))[:num_samples]
            
            generated_images = []
            
            # Generate images in batches to avoid memory issues
            batch_size = 1 if self.device.type == 'cpu' else 4
            for i in range(0, num_samples, batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_size_actual = len(batch_prompts)
                
                self.logger.info(f"Generating batch {i//batch_size + 1}/{(num_samples-1)//batch_size + 1} with {batch_size_actual} images")
                
                # Generate images
                with torch.no_grad():
                    outputs = self.pipeline(
                        batch_prompts,
                        height=self.image_size,
                        width=self.image_size,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps
                    )
                
                # Add generated images to list
                for j in range(batch_size_actual):
                    generated_images.append(outputs.images[j])
            
            self.logger.info(f"Generated {len(generated_images)} synthetic image samples")
            
            return generated_images
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic images: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save the image generator configuration to disk.
        
        Args:
            path: Path to save the configuration to
            
        Raises:
            RuntimeError: If the model has not been loaded yet
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been loaded yet. Call fit() first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save configuration
            config = {
                "model_name": self.model_name,
                "image_size": self.image_size,
                "guidance_scale": self.guidance_scale,
                "num_inference_steps": self.num_inference_steps,
                "model_kwargs": self.model_kwargs
            }
            
            import json
            with open(path, "w") as f:
                json.dump(config, f)
            
            self.logger.info(f"Saved image generator configuration to {path}")
            self.logger.info("Note: The actual model weights are not saved, only the configuration.")
        except Exception as e:
            self.logger.error(f"Error saving image generator configuration: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load an image generator configuration from disk and initialize the model.
        
        Args:
            path: Path to load the configuration from
        """
        try:
            import json
            with open(path, "r") as f:
                config = json.load(f)
            
            self.model_name = config["model_name"]
            self.image_size = config["image_size"]
            self.guidance_scale = config["guidance_scale"]
            self.num_inference_steps = config["num_inference_steps"]
            self.model_kwargs = config["model_kwargs"]
            
            self.logger.info(f"Loaded image generator configuration from {path}")
            
            # Load the model
            self.fit()
            
            self.logger.info(f"Initialized image generator with loaded configuration")
        except Exception as e:
            self.logger.error(f"Error loading image generator configuration: {str(e)}")
            raise
    
    def save_images(self, images: List[Image.Image], output_dir: str, prefix: str = "synthetic_image"):
        """Save generated images to disk.
        
        Args:
            images: List of PIL Image objects
            output_dir: Directory to save images to
            prefix: Prefix for image filenames
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for i, image in enumerate(images):
                image_path = os.path.join(output_dir, f"{prefix}_{i+1}.png")
                image.save(image_path)
            
            self.logger.info(f"Saved {len(images)} images to {output_dir}")
        except Exception as e:
            self.logger.error(f"Error saving images: {str(e)}")
            raise