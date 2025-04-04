"""
Risk Modeling Module

This module provides machine learning models for predicting various
risk factors in insurance underwriting.
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiskModelTrainer:
    """Class for training and evaluating risk prediction models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importances = None
        self.shap_values = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded data as DataFrame
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() == '.json':
            return pd.read_json(file_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            feature_columns: List of feature column names (if None, use all columns except target)
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"Data prepared. Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Train multiple models and return their performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of trained models
        """
        models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            model.fit(X_train, y_train)
            
            cv_results[name] = {
                'model': model,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            }
            
            logger.info(f"{name} - Mean CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        self.models = cv_results
        
        # Determine the best model
        best_model_name = max(cv_results, key=lambda k: cv_results[k]['cv_score_mean'])
        self.best_model = cv_results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} with CV score: {cv_results[best_model_name]['cv_score_mean']:.4f}")
        
        return cv_results
    
    def tune_hyperparameters(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        model_type: str = 'xgboost',
        cv: int = 3
    ) -> Any:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to tune
            cv: Number of cross-validation folds
            
        Returns:
            Best model after hyperparameter tuning
        """
        logger.info(f"Tuning hyperparameters for {model_type}...")
        
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=self.random_state)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return metrics
    
    def calculate_feature_importance(
        self, 
        model: Any, 
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate feature importance from the model.
        
        Args:
            model: Trained model
            feature_names: Names of the features
            
        Returns:
            DataFrame with feature importances
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importances attribute")
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        self.feature_importances = feature_importance
        
        return feature_importance
    
    def calculate_shap_values(
        self, 
        model: Any, 
        X: np.ndarray, 
        feature_names: List[str]
    ) -> None:
        """
        Calculate SHAP values for model explainability.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of the features
        """
        try:
            if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    # For multi-class, take the positive class
                    shap_values = shap_values[1]
                
                self.shap_values = {
                    'values': shap_values,
                    'explainer': explainer,
                    'feature_names': feature_names
                }
                
                logger.info("SHAP values calculated successfully")
            else:
                logger.warning("SHAP values calculation not supported for this model type")
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
    
    def plot_feature_importance(self, output_path: str) -> None:
        """
        Plot feature importance and save to file.
        
        Args:
            output_path: Path to save the plot
        """
        if self.feature_importances is None:
            logger.warning("Feature importances not calculated")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=self.feature_importances.head(15)
        )
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        logger.info(f"Feature importance plot saved to {output_path}")
    
    def plot_shap_summary(self, output_path: str) -> None:
        """
        Plot SHAP summary and save to file.
        
        Args:
            output_path: Path to save the plot
        """
        if self.shap_values is None:
            logger.warning("SHAP values not calculated")
            return
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values['values'], 
            feature_names=self.shap_values['feature_names'],
            show=False
        )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {output_path}")
    
    def save_model(self, model: Any, output_path: str) -> None:
        """
        Save a trained model to a file.
        
        Args:
            model: Trained model
            output_path: Path to save the model
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from a file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def train_and_evaluate(
        self, 
        data_path: str, 
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        model_type: str = 'xgboost',
        tune_hyperparams: bool = True,
        output_dir: str = '../models'
    ) -> Dict[str, Any]:
        """
        End-to-end workflow: load data, train model, evaluate, and save.
        
        Args:
            data_path: Path to the data file
            target_column: Name of the target column
            feature_columns: List of feature column names
            model_type: Type of model to train
            tune_hyperparams: Whether to tune hyperparameters
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with results
        """
        # Load and prepare data
        df = self.load_data(data_path)
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X_train, X_test, y_train, y_test = self.prepare_data(
            df, target_column, feature_columns
        )
        
        # Train model
        if tune_hyperparams:
            model = self.tune_hyperparameters(X_train, y_train, model_type)
        else:
            self.train_models(X_train, y_train)
            model = self.best_model
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Calculate feature importance
        feature_importance = self.calculate_feature_importance(model, feature_columns)
        
        # Calculate SHAP values
        self.calculate_shap_values(model, X_test[:100], feature_columns)  # Use a subset for efficiency
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and plots
        model_filename = f"{model_type}_model.pkl"
        self.save_model(model, os.path.join(output_dir, model_filename))
        
        self.plot_feature_importance(os.path.join(output_dir, 'feature_importance.png'))
        
        if self.shap_values is not None:
            self.plot_shap_summary(os.path.join(output_dir, 'shap_summary.png'))
        
        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': feature_importance
        }


if __name__ == "__main__":
    # Example usage
    trainer = RiskModelTrainer()
    
    # Train and evaluate a model for claim prediction
    results = trainer.train_and_evaluate(
        "../data/processed/underwriting_features.csv",
        target_column="claim_filed",
        model_type="xgboost",
        tune_hyperparams=True,
        output_dir="../models/claim_prediction"
    )
    
    # Train and evaluate a model for risk classification
    results = trainer.train_and_evaluate(
        "../data/processed/risk_features.csv",
        target_column="high_risk",
        model_type="random_forest",
        tune_hyperparams=True,
        output_dir="../models/risk_classification"
    )
