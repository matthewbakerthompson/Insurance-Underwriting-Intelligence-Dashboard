"""
Insurance Underwriting Intelligence Pipeline

This script demonstrates the end-to-end pipeline for insurance underwriting
intelligence, from data processing to model evaluation and visualization.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_processing.third_party_data_integration import DataIntegrator
from src.features.feature_engineering import FeatureEngineer
from src.modeling.risk_models import RiskModelTrainer
from src.evaluation.business_impact import BusinessImpactEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/processed",
        "models",
        "reports/figures",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def process_data():
    """Process raw data and generate features."""
    logger.info("Starting data processing...")
    
    # Process property data
    engineer = FeatureEngineer()
    
    try:
        # Process property data
        logger.info("Processing property data...")
        engineer.process_and_save(
            "data/raw/sample_property_data.csv",
            "data/processed/property_features.csv"
        )
        
        # Load and process business data
        logger.info("Processing business data...")
        business_data = pd.read_json("data/raw/sample_business_data.json")
        
        # Extract businesses from the JSON structure if needed
        if "businesses" in business_data:
            business_data = pd.json_normalize(business_data["businesses"])
        
        # Save to CSV for easier processing
        business_data.to_csv("data/processed/business_data.csv", index=False)
        
        # Engineer features for business data
        engineer.process_and_save(
            "data/processed/business_data.csv",
            "data/processed/business_features.csv"
        )
        
        logger.info("Data processing completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        return False


def train_models():
    """Train and evaluate risk models."""
    logger.info("Starting model training...")
    
    trainer = RiskModelTrainer()
    
    try:
        # Train property risk model
        logger.info("Training property risk model...")
        property_results = trainer.train_and_evaluate(
            "data/processed/property_features.csv",
            target_column="claim_filed",
            model_type="xgboost",
            tune_hyperparams=True,
            output_dir="models/property_risk"
        )
        
        # Train business risk model (assuming we have a target column)
        # In a real scenario, we would have a proper target variable
        # For demonstration, we'll create a synthetic target
        logger.info("Preparing business risk data...")
        business_data = pd.read_csv("data/processed/business_features.csv")
        
        # Create a synthetic target based on risk score
        # In a real scenario, this would be actual historical data
        if "risk_score" in business_data.columns:
            threshold = business_data["risk_score"].median()
            business_data["high_risk"] = (business_data["risk_score"] > threshold).astype(int)
            business_data.to_csv("data/processed/business_risk_features.csv", index=False)
            
            logger.info("Training business risk model...")
            business_results = trainer.train_and_evaluate(
                "data/processed/business_risk_features.csv",
                target_column="high_risk",
                model_type="random_forest",
                tune_hyperparams=True,
                output_dir="models/business_risk"
            )
        
        logger.info("Model training completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return False


def evaluate_business_impact():
    """Evaluate the business impact of the models."""
    logger.info("Starting business impact evaluation...")
    
    evaluator = BusinessImpactEvaluator()
    
    try:
        # Load property data and model predictions
        property_data = pd.read_csv("data/processed/property_features.csv")
        
        # For demonstration, we'll use the actual labels and create synthetic predictions
        # In a real scenario, these would be actual model predictions
        np.random.seed(42)
        y_true = property_data["claim_filed"].values
        
        # Create predictions with 85% accuracy (realistic but not perfect)
        accuracy = 0.85
        n_samples = len(y_true)
        n_correct = int(n_samples * accuracy)
        n_incorrect = n_samples - n_correct
        
        # Start with all correct predictions
        y_pred = y_true.copy()
        
        # Randomly flip some predictions to achieve desired accuracy
        flip_indices = np.random.choice(
            np.arange(n_samples), 
            size=n_incorrect, 
            replace=False
        )
        for idx in flip_indices:
            y_pred[idx] = 1 - y_pred[idx]
        
        # Calculate cost-benefit analysis
        logger.info("Calculating cost-benefit analysis...")
        cost_matrix = {
            'tp_benefit': 5000,  # Benefit from correctly identifying a claim
            'tn_benefit': 500,   # Benefit from correctly identifying a non-claim
            'fp_cost': 1000,     # Cost of falsely predicting a claim
            'fn_cost': 8000      # Cost of missing a claim
        }
        
        evaluator.calculate_cost_benefit(y_true, y_pred, cost_matrix)
        evaluator.plot_cost_benefit_analysis("reports/figures/cost_benefit_analysis.png")
        
        # Calculate efficiency gains
        logger.info("Calculating efficiency gains...")
        evaluator.calculate_efficiency_gains(
            manual_time_per_case=2.5,      # hours
            automated_time_per_case=0.5,   # hours
            num_cases=1000,
            hourly_cost=75                 # dollars
        )
        evaluator.plot_efficiency_gains("reports/figures/efficiency_gains.png")
        
        # Calculate risk reduction
        logger.info("Calculating risk reduction...")
        evaluator.calculate_risk_reduction(
            baseline_loss_ratio=0.65,
            predicted_loss_ratio=0.58,
            total_premium=10000000
        )
        evaluator.plot_risk_reduction("reports/figures/risk_reduction.png")
        
        # Generate impact report
        logger.info("Generating impact report...")
        evaluator.generate_impact_report("reports/business_impact_report.json")
        
        logger.info("Business impact evaluation completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in business impact evaluation: {e}")
        return False


def run_dashboard():
    """Run the Streamlit dashboard."""
    logger.info("Starting dashboard...")
    
    try:
        # Use subprocess to run the Streamlit app
        import subprocess
        
        # Run the Streamlit app
        dashboard_process = subprocess.Popen(
            ["streamlit", "run", "src/dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print the URL
        logger.info("Dashboard started. Access it at http://localhost:8501")
        
        # In a real application, we might want to handle the process differently
        # For demonstration purposes, we'll just return
        return True
    
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False


def main():
    """Run the complete pipeline."""
    logger.info("Starting Insurance Underwriting Intelligence Pipeline")
    
    # Create necessary directories
    create_directories()
    
    # Process data
    if process_data():
        logger.info("Data processing completed successfully")
    else:
        logger.error("Data processing failed")
        return
    
    # Train models
    if train_models():
        logger.info("Model training completed successfully")
    else:
        logger.error("Model training failed")
        return
    
    # Evaluate business impact
    if evaluate_business_impact():
        logger.info("Business impact evaluation completed successfully")
    else:
        logger.error("Business impact evaluation failed")
        return
    
    # Run dashboard
    if run_dashboard():
        logger.info("Dashboard started successfully")
    else:
        logger.error("Dashboard startup failed")
        return
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
