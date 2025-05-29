import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.metrics import mutual_info_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

class DataEvaluator:
    """Utility class for evaluating synthetic data quality and similarity to original data."""
    
    def __init__(self, original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Initialize the data evaluator.
        
        Args:
            original_data: Original input DataFrame
            synthetic_data: Synthetic DataFrame for evaluation
        """
        self.logger = logging.getLogger(__name__)
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.logger.info(f"Initialized data evaluator for data with shapes {original_data.shape} and {synthetic_data.shape}")
    
    def update_synthetic_data(self, synthetic_data: pd.DataFrame) -> None:
        """Update the synthetic data reference.
        
        Args:
            synthetic_data: New synthetic DataFrame for evaluation
        """
        self.synthetic_data = synthetic_data
        self.logger.info(f"Updated synthetic data with shape {synthetic_data.shape}")
    
    def evaluate_statistical_similarity(self) -> Dict[str, Dict[str, float]]:
        """Evaluate statistical similarity between original and synthetic data.
        
        Returns:
            Dictionary with statistical similarity metrics by column and overall
        """
        self.logger.info("Evaluating statistical similarity")
        
        # Initialize results dictionary
        results = {
            "column_metrics": {},
            "overall": {}
        }
        
        # Get common columns
        common_columns = [col for col in self.original_data.columns if col in self.synthetic_data.columns]
        if not common_columns:
            self.logger.error("No common columns found between original and synthetic data")
            return {"error": "No common columns found"}
        
        # Calculate column-level metrics
        for column in common_columns:
            column_metrics = {}
            
            # Get original and synthetic values for the column
            orig_values = self.original_data[column]
            syn_values = self.synthetic_data[column]
            
            # Skip columns with different data types
            if orig_values.dtype != syn_values.dtype:
                self.logger.warning(f"Column '{column}' has different data types in original and synthetic data")
                continue
            
            # Calculate metrics based on data type
            if pd.api.types.is_numeric_dtype(orig_values):
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_pval = ks_2samp(orig_values.dropna(), syn_values.dropna())
                    column_metrics["ks_statistic"] = ks_stat
                    column_metrics["ks_pvalue"] = ks_pval
                except Exception as e:
                    self.logger.warning(f"Failed to calculate KS test for column '{column}': {str(e)}")
                
                # Wasserstein distance (Earth Mover's Distance)
                try:
                    w_dist = wasserstein_distance(orig_values.dropna(), syn_values.dropna())
                    column_metrics["wasserstein_distance"] = w_dist
                    # Normalize Wasserstein distance
                    orig_range = orig_values.max() - orig_values.min()
                    if orig_range > 0:
                        column_metrics["wasserstein_normalized"] = w_dist / orig_range
                except Exception as e:
                    self.logger.warning(f"Failed to calculate Wasserstein distance for column '{column}': {str(e)}")
                
                # Mean and standard deviation differences
                orig_mean = orig_values.mean()
                syn_mean = syn_values.mean()
                column_metrics["mean_difference"] = abs(orig_mean - syn_mean)
                column_metrics["mean_difference_percentage"] = abs(orig_mean - syn_mean) / abs(orig_mean) * 100 if orig_mean != 0 else 0
                
                orig_std = orig_values.std()
                syn_std = syn_values.std()
                column_metrics["std_difference"] = abs(orig_std - syn_std)
                column_metrics["std_difference_percentage"] = abs(orig_std - syn_std) / abs(orig_std) * 100 if orig_std != 0 else 0
                
            elif pd.api.types.is_object_dtype(orig_values) or pd.api.types.is_categorical_dtype(orig_values):
                # Calculate distribution similarity for categorical data
                try:
                    # Get value counts and normalize
                    orig_counts = orig_values.value_counts(normalize=True).fillna(0)
                    syn_counts = syn_values.value_counts(normalize=True).fillna(0)
                    
                    # Align categories
                    all_categories = pd.Index(set(orig_counts.index) | set(syn_counts.index))
                    orig_counts = orig_counts.reindex(all_categories, fill_value=0)
                    syn_counts = syn_counts.reindex(all_categories, fill_value=0)
                    
                    # Calculate Jensen-Shannon divergence
                    js_div = jensenshannon(orig_counts.values, syn_counts.values)
                    column_metrics["jensen_shannon_divergence"] = js_div
                    column_metrics["jensen_shannon_similarity"] = 1 - js_div
                    
                    # Calculate category overlap
                    orig_categories = set(orig_values.dropna().unique())
                    syn_categories = set(syn_values.dropna().unique())
                    overlap = len(orig_categories.intersection(syn_categories))
                    column_metrics["category_overlap"] = overlap / len(orig_categories) if orig_categories else 0
                    
                except Exception as e:
                    self.logger.warning(f"Failed to calculate categorical metrics for column '{column}': {str(e)}")
            
            # Add column metrics to results
            results["column_metrics"][column] = column_metrics
        
        # Calculate overall metrics
        
        # 1. Average similarity across columns
        similarity_scores = []
        for column, metrics in results["column_metrics"].items():
            if pd.api.types.is_numeric_dtype(self.original_data[column]):
                # For numeric columns, use 1 - normalized Wasserstein distance as similarity
                if "wasserstein_normalized" in metrics:
                    similarity = max(0, 1 - min(metrics["wasserstein_normalized"], 1))
                    similarity_scores.append(similarity)
            else:
                # For categorical columns, use Jensen-Shannon similarity
                if "jensen_shannon_similarity" in metrics:
                    similarity_scores.append(metrics["jensen_shannon_similarity"])
        
        if similarity_scores:
            results["overall"]["average_similarity"] = np.mean(similarity_scores)
        
        # 2. Correlation matrix similarity
        try:
            # Get numeric columns that are common to both datasets
            numeric_cols = [col for col in common_columns if pd.api.types.is_numeric_dtype(self.original_data[col])]
            
            if len(numeric_cols) >= 2:  # Need at least 2 columns for correlation
                orig_corr = self.original_data[numeric_cols].corr().fillna(0)
                syn_corr = self.synthetic_data[numeric_cols].corr().fillna(0)
                
                # Calculate Frobenius norm of the difference
                corr_diff = np.linalg.norm(orig_corr.values - syn_corr.values)
                max_diff = np.sqrt(2 * len(numeric_cols) * len(numeric_cols))  # Maximum possible difference
                results["overall"]["correlation_similarity"] = 1 - (corr_diff / max_diff)
        except Exception as e:
            self.logger.warning(f"Failed to calculate correlation similarity: {str(e)}")
        
        # 3. Overall statistical similarity score (weighted average of all metrics)
        if "average_similarity" in results["overall"] and "correlation_similarity" in results["overall"]:
            results["overall"]["statistical_similarity_score"] = 0.7 * results["overall"]["average_similarity"] + \
                                                              0.3 * results["overall"]["correlation_similarity"]
        elif "average_similarity" in results["overall"]:
            results["overall"]["statistical_similarity_score"] = results["overall"]["average_similarity"]
        
        self.logger.info("Statistical similarity evaluation completed")
        return results
    
    def evaluate_machine_learning_utility(self, target_column: str, test_size: float = 0.2, 
                                         random_state: int = 42) -> Dict[str, Any]:
        """Evaluate machine learning utility by training models on original and synthetic data.
        
        Args:
            target_column: Target column for prediction
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with machine learning utility metrics
        """
        self.logger.info(f"Evaluating machine learning utility with target column '{target_column}'")
        
        # Check if target column exists in both datasets
        if target_column not in self.original_data.columns or target_column not in self.synthetic_data.columns:
            self.logger.error(f"Target column '{target_column}' not found in both datasets")
            return {"error": f"Target column '{target_column}' not found in both datasets"}
        
        # Get feature columns (excluding target)
        feature_cols = [col for col in self.original_data.columns if col != target_column and 
                       col in self.synthetic_data.columns]
        
        # Prepare data
        X_orig = self.original_data[feature_cols]
        y_orig = self.original_data[target_column]
        X_syn = self.synthetic_data[feature_cols]
        y_syn = self.synthetic_data[target_column]
        
        # Handle non-numeric columns
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(X_orig[col]):
                # Convert to categorical codes
                X_orig[col] = X_orig[col].astype('category').cat.codes
                X_syn[col] = X_syn[col].astype('category').cat.codes
        
        if not pd.api.types.is_numeric_dtype(y_orig):
            y_orig = y_orig.astype('category').cat.codes
            y_syn = y_syn.astype('category').cat.codes
        
        # Split original data into train and test sets
        X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
            X_orig, y_orig, test_size=test_size, random_state=random_state
        )
        
        # Initialize results dictionary
        results = {}
        
        try:
            # Train model on original data
            model_orig = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model_orig.fit(X_orig_train, y_orig_train)
            orig_score = model_orig.score(X_orig_test, y_orig_test)
            results["original_model_accuracy"] = orig_score
            
            # Train model on synthetic data
            model_syn = RandomForestClassifier(n_estimators=100, random_state=random_state)
            model_syn.fit(X_syn, y_syn)
            syn_score = model_syn.score(X_orig_test, y_orig_test)
            results["synthetic_model_accuracy"] = syn_score
            
            # Calculate relative performance
            results["relative_performance"] = syn_score / orig_score if orig_score > 0 else 0
            
            # Feature importance similarity
            orig_importance = model_orig.feature_importances_
            syn_importance = model_syn.feature_importances_
            importance_similarity = 1 - np.mean(np.abs(orig_importance - syn_importance))
            results["feature_importance_similarity"] = importance_similarity
            
            # Detailed classification report for synthetic model
            y_pred = model_syn.predict(X_orig_test)
            results["classification_report"] = classification_report(y_orig_test, y_pred, output_dict=True)
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate machine learning utility: {str(e)}")
            results["error"] = str(e)
        
        self.logger.info("Machine learning utility evaluation completed")
        return results
    
    def evaluate_privacy(self, distance_threshold: float = 0.1) -> Dict[str, Any]:
        """Evaluate privacy risks in synthetic data.
        
        Args:
            distance_threshold: Threshold for distance-based privacy metrics
            
        Returns:
            Dictionary with privacy evaluation metrics
        """
        self.logger.info("Evaluating privacy risks")
        
        # Initialize results dictionary
        results = {
            "membership_inference": {},
            "attribute_disclosure": {},
            "distance_based": {}
        }
        
        # Get common columns
        common_columns = [col for col in self.original_data.columns if col in self.synthetic_data.columns]
        
        # 1. Distance-based privacy metrics
        try:
            # Prepare data (numeric columns only)
            numeric_cols = [col for col in common_columns if pd.api.types.is_numeric_dtype(self.original_data[col])]
            
            if numeric_cols:
                # Normalize data
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                orig_scaled = scaler.fit_transform(self.original_data[numeric_cols])
                syn_scaled = scaler.transform(self.synthetic_data[numeric_cols])
                
                # Calculate minimum distances between original and synthetic records
                from sklearn.metrics import pairwise_distances
                distances = pairwise_distances(orig_scaled, syn_scaled, metric='euclidean')
                min_distances = np.min(distances, axis=1)
                
                # Calculate privacy metrics
                results["distance_based"]["mean_minimum_distance"] = float(np.mean(min_distances))
                results["distance_based"]["median_minimum_distance"] = float(np.median(min_distances))
                results["distance_based"]["min_minimum_distance"] = float(np.min(min_distances))
                
                # Calculate proportion of records with distance below threshold
                below_threshold = np.sum(min_distances < distance_threshold) / len(min_distances)
                results["distance_based"]["proportion_below_threshold"] = float(below_threshold)
                
                # Privacy risk score (inverse of mean minimum distance, normalized to 0-1)
                risk_score = 1 / (1 + np.mean(min_distances))
                results["distance_based"]["privacy_risk_score"] = float(risk_score)
        except Exception as e:
            self.logger.warning(f"Failed to calculate distance-based privacy metrics: {str(e)}")
        
        # 2. Membership inference risk
        try:
            # Train a classifier to distinguish between original and synthetic data
            combined_data = pd.concat([self.original_data[common_columns], self.synthetic_data[common_columns]])
            labels = np.concatenate([np.ones(len(self.original_data)), np.zeros(len(self.synthetic_data))])
            
            # Handle non-numeric columns
            for col in combined_data.columns:
                if not pd.api.types.is_numeric_dtype(combined_data[col]):
                    combined_data[col] = combined_data[col].astype('category').cat.codes
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.3, random_state=42)
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate classifier
            accuracy = clf.score(X_test, y_test)
            results["membership_inference"]["classifier_accuracy"] = float(accuracy)
            
            # Calculate membership inference risk (0.5 = no risk, 1.0 = high risk)
            # Normalize from 0.5-1.0 range to 0.0-1.0 range
            risk_score = (accuracy - 0.5) * 2 if accuracy > 0.5 else 0
            results["membership_inference"]["risk_score"] = float(risk_score)
        except Exception as e:
            self.logger.warning(f"Failed to calculate membership inference risk: {str(e)}")
        
        # 3. Attribute disclosure risk (for categorical columns)
        try:
            categorical_cols = [col for col in common_columns if 
                              pd.api.types.is_object_dtype(self.original_data[col]) or 
                              pd.api.types.is_categorical_dtype(self.original_data[col])]            
            
            if categorical_cols:
                # Calculate mutual information between columns in synthetic data
                mutual_info = {}
                for i, col1 in enumerate(categorical_cols):
                    for col2 in categorical_cols[i+1:]:
                        # Convert to category codes
                        orig_col1 = self.original_data[col1].astype('category').cat.codes
                        orig_col2 = self.original_data[col2].astype('category').cat.codes
                        syn_col1 = self.synthetic_data[col1].astype('category').cat.codes
                        syn_col2 = self.synthetic_data[col2].astype('category').cat.codes
                        
                        # Calculate mutual information
                        orig_mi = mutual_info_score(orig_col1, orig_col2)
                        syn_mi = mutual_info_score(syn_col1, syn_col2)
                        
                        # Calculate relative difference
                        if orig_mi > 0:
                            rel_diff = abs(orig_mi - syn_mi) / orig_mi
                            mutual_info[f"{col1}_{col2}"] = rel_diff
                
                if mutual_info:
                    # Average relative difference in mutual information
                    avg_rel_diff = np.mean(list(mutual_info.values()))
                    results["attribute_disclosure"]["avg_mutual_info_difference"] = float(avg_rel_diff)
                    
                    # Attribute disclosure risk score (lower difference = higher risk)
                    risk_score = 1 / (1 + avg_rel_diff)
                    results["attribute_disclosure"]["risk_score"] = float(risk_score)
        except Exception as e:
            self.logger.warning(f"Failed to calculate attribute disclosure risk: {str(e)}")
        
        # 4. Overall privacy risk score (weighted average of individual risk scores)
        risk_scores = []
        if "privacy_risk_score" in results["distance_based"]:
            risk_scores.append((results["distance_based"]["privacy_risk_score"], 0.4))  # 40% weight
        if "risk_score" in results["membership_inference"]:
            risk_scores.append((results["membership_inference"]["risk_score"], 0.4))  # 40% weight
        if "risk_score" in results["attribute_disclosure"]:
            risk_scores.append((results["attribute_disclosure"]["risk_score"], 0.2))  # 20% weight
        
        if risk_scores:
            weighted_sum = sum(score * weight for score, weight in risk_scores)
            total_weight = sum(weight for _, weight in risk_scores)
            results["overall_privacy_risk_score"] = float(weighted_sum / total_weight)
        
        self.logger.info("Privacy evaluation completed")
        return results
    
    def get_overall_quality_score(self) -> Dict[str, float]:
        """Calculate an overall quality score for the synthetic data.
        
        Returns:
            Dictionary with overall quality scores
        """
        self.logger.info("Calculating overall quality score")
        
        # Get individual evaluation results
        statistical_results = self.evaluate_statistical_similarity()
        
        # Try to get a target column for ML utility evaluation
        target_column = None
        for col in self.original_data.columns:
            if col in self.synthetic_data.columns:
                if (pd.api.types.is_object_dtype(self.original_data[col]) or 
                    pd.api.types.is_categorical_dtype(self.original_data[col])):
                    target_column = col
                    break
        
        ml_results = {}
        if target_column:
            ml_results = self.evaluate_machine_learning_utility(target_column)
        
        privacy_results = self.evaluate_privacy()
        
        # Initialize scores dictionary
        scores = {}
        
        # 1. Statistical similarity score (40% of overall score)
        if "overall" in statistical_results and "statistical_similarity_score" in statistical_results["overall"]:
            scores["statistical_similarity"] = statistical_results["overall"]["statistical_similarity_score"]
        else:
            scores["statistical_similarity"] = 0.0
        
        # 2. Machine learning utility score (30% of overall score)
        if "relative_performance" in ml_results:
            # Cap at 1.0 (synthetic can't be better than original for scoring purposes)
            scores["ml_utility"] = min(ml_results["relative_performance"], 1.0)
        else:
            scores["ml_utility"] = 0.0
        
        # 3. Privacy score (30% of overall score)
        # For privacy, lower risk score is better
        if "overall_privacy_risk_score" in privacy_results:
            # Convert risk score to privacy score (1 - risk)
            scores["privacy"] = 1.0 - privacy_results["overall_privacy_risk_score"]
        else:
            scores["privacy"] = 0.0
        
        # Calculate overall score (weighted average)
        scores["overall"] = 0.4 * scores["statistical_similarity"] + \
                          0.3 * scores["ml_utility"] + \
                          0.3 * scores["privacy"]
        
        # Add letter grade
        scores["grade"] = self._get_letter_grade(scores["overall"])
        
        self.logger.info(f"Overall quality score: {scores['overall']:.2f} ({scores['grade']})")
        return scores
    
    def _get_letter_grade(self, score: float) -> str:
        """Convert numerical score to letter grade.
        
        Args:
            score: Numerical score (0-1)
            
        Returns:
            Letter grade (A+ to F)
        """
        if score >= 0.97:
            return "A+"
        elif score >= 0.93:
            return "A"
        elif score >= 0.90:
            return "A-"
        elif score >= 0.87:
            return "B+"
        elif score >= 0.83:
            return "B"
        elif score >= 0.80:
            return "B-"
        elif score >= 0.77:
            return "C+"
        elif score >= 0.73:
            return "C"
        elif score >= 0.70:
            return "C-"
        elif score >= 0.67:
            return "D+"
        elif score >= 0.63:
            return "D"
        elif score >= 0.60:
            return "D-"
        else:
            return "F"