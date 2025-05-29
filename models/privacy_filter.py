import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

class PrivacyFilter:
    """Privacy filter for ensuring synthetic data is privacy-preserving."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 0.001, k_anonymity: int = 5):
        """Initialize the privacy filter.
        
        Args:
            epsilon: Privacy budget for differential privacy (lower = more private)
            delta: Probability of privacy breach (lower = more private)
            k_anonymity: k value for k-anonymity (higher = more private)
        """
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon
        self.delta = delta
        self.k_anonymity = k_anonymity
        self.logger.info(f"Initialized privacy filter with epsilon={epsilon}, delta={delta}, k_anonymity={k_anonymity}")
    
    def apply(self, synthetic_data, real_data=None):
        """Apply privacy filtering to synthetic data.
        
        Args:
            synthetic_data: Synthetic data to filter
            real_data: Optional real data for reference
            
        Returns:
            Privacy-filtered synthetic data
        """
        self.logger.info(f"Applying privacy filtering to synthetic data with shape {synthetic_data.shape}")
        
        # Apply differential privacy if epsilon is set
        if self.epsilon is not None and self.epsilon > 0:
            synthetic_data = self._apply_differential_privacy(synthetic_data)
        
        # Apply k-anonymity if k_anonymity is set
        if self.k_anonymity is not None and self.k_anonymity > 1:
            synthetic_data = self._apply_k_anonymity(synthetic_data)
        
        # Apply membership inference protection if real_data is provided
        if real_data is not None:
            synthetic_data = self._protect_against_membership_inference(synthetic_data, real_data)
        
        self.logger.info(f"Privacy filtering completed, resulting shape: {synthetic_data.shape}")
        return synthetic_data
    
    def _apply_differential_privacy(self, data):
        """Apply differential privacy to the data.
        
        Args:
            data: Data to apply differential privacy to
            
        Returns:
            Data with differential privacy applied
        """
        try:
            from diffprivlib.models import GaussianNB
            from diffprivlib.mechanisms import Laplace
            
            self.logger.info(f"Applying differential privacy with epsilon={self.epsilon}, delta={self.delta}")
            
            # Make a copy of the data
            dp_data = data.copy()
            
            # Apply Laplace noise to numerical columns
            for column in dp_data.select_dtypes(include=['number']).columns:
                # Calculate sensitivity based on column range
                sensitivity = (dp_data[column].max() - dp_data[column].min()) / 10
                
                # Create Laplace mechanism
                mech = Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
                
                # Add noise to the column
                dp_data[column] = dp_data[column].apply(lambda x: mech.randomise(x))
            
            self.logger.info("Differential privacy applied successfully")
            return dp_data
            
        except ImportError:
            self.logger.warning("diffprivlib package not found. Skipping differential privacy.")
            return data
        except Exception as e:
            self.logger.error(f"Error applying differential privacy: {str(e)}")
            return data
    
    def _apply_k_anonymity(self, data):
        """Apply k-anonymity to the data.
        
        Args:
            data: Data to apply k-anonymity to
            
        Returns:
            Data with k-anonymity applied
        """
        self.logger.info(f"Applying k-anonymity with k={self.k_anonymity}")
        
        # Make a copy of the data
        k_anon_data = data.copy()
        
        # Identify potential quasi-identifiers (categorical columns with low cardinality)
        quasi_identifiers = []
        for column in k_anon_data.columns:
            if k_anon_data[column].dtype == 'object' or k_anon_data[column].dtype == 'category':
                if len(k_anon_data[column].unique()) < len(k_anon_data) / 10:  # Heuristic
                    quasi_identifiers.append(column)
        
        if not quasi_identifiers:
            self.logger.info("No suitable quasi-identifiers found for k-anonymity")
            return data
        
        self.logger.info(f"Identified {len(quasi_identifiers)} quasi-identifiers: {quasi_identifiers}")
        
        # Group by quasi-identifiers and check group sizes
        grouped = k_anon_data.groupby(quasi_identifiers).size().reset_index(name='count')
        small_groups = grouped[grouped['count'] < self.k_anonymity]
        
        if len(small_groups) == 0:
            self.logger.info("Data already satisfies k-anonymity")
            return data
        
        # Remove records from small groups
        for _, row in small_groups.iterrows():
            mask = pd.Series(True, index=k_anon_data.index)
            for qi in quasi_identifiers:
                mask = mask & (k_anon_data[qi] == row[qi])
            k_anon_data = k_anon_data[~mask]
        
        self.logger.info(f"Removed {len(data) - len(k_anon_data)} records to satisfy k-anonymity")
        return k_anon_data
    
    def _protect_against_membership_inference(self, synthetic_data, real_data):
        """Protect against membership inference attacks.
        
        Args:
            synthetic_data: Synthetic data to protect
            real_data: Real data for reference
            
        Returns:
            Protected synthetic data
        """
        self.logger.info("Applying membership inference protection")
        
        # Make a copy of the synthetic data
        protected_data = synthetic_data.copy()
        
        try:
            # Calculate minimum distance between synthetic and real records
            min_distances = self._calculate_min_distances(protected_data, real_data)
            
            # Identify synthetic records that are too close to real records
            threshold = np.percentile(min_distances, 10)  # Bottom 10% are too close
            too_close_indices = np.where(min_distances < threshold)[0]
            
            if len(too_close_indices) > 0:
                self.logger.info(f"Found {len(too_close_indices)} synthetic records too close to real data")
                
                # Add small noise to records that are too close
                for idx in too_close_indices:
                    for column in protected_data.select_dtypes(include=['number']).columns:
                        # Add small random noise
                        std = protected_data[column].std() * 0.1
                        protected_data.loc[protected_data.index[idx], column] += np.random.normal(0, std)
            
            self.logger.info("Membership inference protection applied successfully")
            return protected_data
            
        except Exception as e:
            self.logger.error(f"Error applying membership inference protection: {str(e)}")
            return synthetic_data
    
    def _calculate_min_distances(self, synthetic_data, real_data):
        """Calculate minimum distances between synthetic and real records.
        
        Args:
            synthetic_data: Synthetic data
            real_data: Real data
            
        Returns:
            Array of minimum distances for each synthetic record
        """
        # Select only numeric columns for distance calculation
        numeric_cols = synthetic_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns found for distance calculation")
            return np.ones(len(synthetic_data))  # Return dummy distances
        
        # Normalize data for distance calculation
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_data[numeric_cols])
        synthetic_scaled = scaler.transform(synthetic_data[numeric_cols])
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(synthetic_scaled, real_scaled)
        
        # Get minimum distance for each synthetic record
        min_distances = np.min(distances, axis=1)
        
        return min_distances
    
    def evaluate_privacy(self, synthetic_data, real_data):
        """Evaluate privacy metrics for synthetic data.
        
        Args:
            synthetic_data: Synthetic data to evaluate
            real_data: Real data for reference
            
        Returns:
            Dictionary of privacy metrics
        """
        self.logger.info("Evaluating privacy metrics")
        
        privacy_metrics = {}
        
        try:
            # Calculate minimum distances for membership inference risk
            min_distances = self._calculate_min_distances(synthetic_data, real_data)
            privacy_metrics['nn_distance'] = float(np.mean(min_distances))
            
            # Train a simple classifier to detect membership (simplified)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            
            # Select numeric columns
            numeric_cols = synthetic_data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) > 0:
                # Create a combined dataset with labels
                real_sample = real_data[numeric_cols].sample(min(len(real_data), 1000))
                synthetic_sample = synthetic_data[numeric_cols].sample(min(len(synthetic_data), 1000))
                
                X = pd.concat([real_sample, synthetic_sample])
                y = np.concatenate([np.ones(len(real_sample)), np.zeros(len(synthetic_sample))])
                
                # Train a classifier
                clf = RandomForestClassifier(n_estimators=10)
                clf.fit(X, y)
                
                # Get predictions
                y_pred = clf.predict_proba(X)[:, 1]
                
                # Calculate AUC
                auc = roc_auc_score(y, y_pred)
                privacy_metrics['membership_inference_auc'] = float(auc)
                
                # Interpret AUC (closer to 0.5 is better for privacy)
                privacy_level = "High" if auc < 0.6 else "Medium" if auc < 0.75 else "Low"
                privacy_metrics['privacy_level'] = privacy_level
            else:
                privacy_metrics['membership_inference_auc'] = 0.5  # Default when no numeric columns
                privacy_metrics['privacy_level'] = "Unknown"
            
            self.logger.info(f"Privacy evaluation completed: {privacy_metrics}")
            return privacy_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating privacy: {str(e)}")
            return {'error': str(e)}