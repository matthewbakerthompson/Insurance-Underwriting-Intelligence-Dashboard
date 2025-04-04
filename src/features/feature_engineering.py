"""
Feature Engineering Module

This module provides functionality for transforming raw data into features
suitable for machine learning models in insurance underwriting.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for engineering features from raw insurance data."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
    
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
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.json_normalize(data)
        elif ext.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def identify_feature_types(self, df: pd.DataFrame, categorical_threshold: int = 10) -> None:
        """
        Identify numeric and categorical features in the DataFrame.
        
        Args:
            df: Input DataFrame
            categorical_threshold: Maximum number of unique values for a column to be considered categorical
        """
        # Identify numeric and categorical features
        numeric_features = []
        categorical_features = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= categorical_threshold:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        logger.info(f"Identified {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
    
    def create_preprocessor(self) -> None:
        """Create a preprocessing pipeline for numeric and categorical features."""
        if self.numeric_features is None or self.categorical_features is None:
            raise ValueError("Feature types not identified. Call identify_feature_types first.")
        
        # Create preprocessing pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        logger.info("Created preprocessing pipeline")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df_engineered = df.copy()
        
        # Add derived features for property data
        if 'property_age' in df_engineered.columns:
            df_engineered['property_age_squared'] = df_engineered['property_age'] ** 2
            df_engineered['is_new_property'] = (df_engineered['property_age'] < 5).astype(int)
        
        if 'property_value' in df_engineered.columns and 'property_size' in df_engineered.columns:
            df_engineered['value_per_sqft'] = df_engineered['property_value'] / df_engineered['property_size']
        
        # Add derived features for business data
        if 'revenue' in df_engineered.columns and 'employees' in df_engineered.columns:
            df_engineered['revenue_per_employee'] = df_engineered['revenue'] / df_engineered['employees']
        
        if 'claims_count' in df_engineered.columns and 'policy_years' in df_engineered.columns:
            df_engineered['claims_per_year'] = df_engineered['claims_count'] / df_engineered['policy_years']
        
        # Add derived features for risk data
        if 'flood_risk_score' in df_engineered.columns and 'fire_risk_score' in df_engineered.columns:
            df_engineered['combined_risk_score'] = (
                df_engineered['flood_risk_score'] + df_engineered['fire_risk_score']
            ) / 2
        
        # Update feature lists
        self.identify_feature_types(df_engineered)
        
        logger.info(f"Engineered features. New shape: {df_engineered.shape}")
        
        return df_engineered
    
    def create_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Create time-based features from a date column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date column
            
        Returns:
            DataFrame with time features added
        """
        # Create a copy to avoid modifying the original
        df_time = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df_time[date_column]):
            df_time[date_column] = pd.to_datetime(df_time[date_column])
        
        # Extract date components
        df_time[f'{date_column}_year'] = df_time[date_column].dt.year
        df_time[f'{date_column}_month'] = df_time[date_column].dt.month
        df_time[f'{date_column}_day'] = df_time[date_column].dt.day
        df_time[f'{date_column}_dayofweek'] = df_time[date_column].dt.dayofweek
        df_time[f'{date_column}_quarter'] = df_time[date_column].dt.quarter
        df_time[f'{date_column}_is_month_end'] = df_time[date_column].dt.is_month_end.astype(int)
        
        logger.info(f"Created time features from {date_column}")
        
        return df_time
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Create interaction features between pairs of numeric features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples containing pairs of feature names to interact
            
        Returns:
            DataFrame with interaction features added
        """
        # Create a copy to avoid modifying the original
        df_interact = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_interact.columns and feat2 in df_interact.columns:
                if pd.api.types.is_numeric_dtype(df_interact[feat1]) and pd.api.types.is_numeric_dtype(df_interact[feat2]):
                    # Product interaction
                    df_interact[f'{feat1}_x_{feat2}'] = df_interact[feat1] * df_interact[feat2]
                    
                    # Sum interaction
                    df_interact[f'{feat1}_plus_{feat2}'] = df_interact[feat1] + df_interact[feat2]
                    
                    # Ratio interaction (with handling for division by zero)
                    df_interact[f'{feat1}_div_{feat2}'] = df_interact[feat1] / df_interact[feat2].replace(0, np.nan)
                    df_interact[f'{feat1}_div_{feat2}'].fillna(0, inplace=True)
        
        logger.info(f"Created interaction features for {len(feature_pairs)} feature pairs")
        
        return df_interact
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor to the data and transform it.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features as numpy array
        """
        if self.preprocessor is None:
            self.identify_feature_types(df)
            self.create_preprocessor()
        
        return self.preprocessor.fit_transform(df)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features as numpy array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(df)
    
    def process_and_save(self, input_path: str, output_path: str) -> None:
        """
        Load, process, and save data with engineered features.
        
        Args:
            input_path: Path to the input data file
            output_path: Path to save the processed data
        """
        # Load data
        df = self.load_data(input_path)
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Create time features if date columns exist
        date_columns = [col for col in df_engineered.columns if 'date' in col.lower()]
        for date_col in date_columns:
            df_engineered = self.create_time_features(df_engineered, date_col)
        
        # Create interaction features for numeric columns
        numeric_cols = [col for col in df_engineered.columns if pd.api.types.is_numeric_dtype(df_engineered[col])]
        if len(numeric_cols) >= 2:
            # Create some example feature pairs (first few combinations)
            feature_pairs = [(numeric_cols[i], numeric_cols[j]) 
                            for i in range(min(3, len(numeric_cols))) 
                            for j in range(i+1, min(4, len(numeric_cols)))]
            df_engineered = self.create_interaction_features(df_engineered, feature_pairs)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed data
        _, ext = os.path.splitext(output_path)
        if ext.lower() == '.csv':
            df_engineered.to_csv(output_path, index=False)
        elif ext.lower() == '.json':
            df_engineered.to_json(output_path, orient='records', indent=2)
        elif ext.lower() in ['.xlsx', '.xls']:
            df_engineered.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {ext}")
        
        logger.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Process property data
    engineer.process_and_save(
        "../data/raw/property_data.csv",
        "../data/processed/property_features.csv"
    )
    
    # Process business data
    engineer.process_and_save(
        "../data/raw/business_data.json",
        "../data/processed/business_features.json"
    )
