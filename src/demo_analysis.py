"""
Insurance Underwriting Data Analysis Demo

This script demonstrates key components of the insurance underwriting
intelligence project with a focus on data analysis and visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Create necessary directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

print("Insurance Underwriting Data Analysis Demo")
print("----------------------------------------")

# Load the property data
print("\nLoading property data...")
try:
    property_data = pd.read_csv("data/raw/sample_property_data.csv")
    print(f"Loaded {len(property_data)} property records")
    print(f"Columns: {', '.join(property_data.columns)}")
    
    # Display basic statistics
    print("\nProperty Data Summary:")
    print(f"Property types: {property_data['property_type'].value_counts().to_dict()}")
    print(f"Average property value: ${property_data['property_value'].mean():.2f}")
    print(f"Average property age: {property_data['property_age'].mean():.2f} years")
    print(f"Claim rate: {property_data['claim_filed'].mean():.2%}")
    
    # Create a processed version with only numeric features for modeling
    print("\nProcessing property data for modeling...")
    model_data = property_data.copy()
    
    # Convert categorical features to dummy variables
    model_data = pd.get_dummies(model_data, columns=['property_type', 'location_risk'], drop_first=False)
    
    # Remove non-numeric identifier column
    model_data = model_data.drop('property_id', axis=1)
    
    # Save processed data
    model_data.to_csv("data/processed/property_model_data.csv", index=False)
    print(f"Saved processed data with {model_data.shape[1]} features")
    
    # Simple feature engineering
    print("\nPerforming feature engineering...")
    model_data['combined_risk'] = (model_data['flood_risk_score'] + model_data['fire_risk_score']) / 2
    model_data['value_per_sqft'] = model_data['property_value'] / model_data['property_size']
    model_data['is_old_property'] = (model_data['property_age'] > 30).astype(int)
    
    # Analyze correlations
    print("\nAnalyzing feature correlations...")
    numeric_cols = ['property_value', 'property_age', 'property_size', 
                    'flood_risk_score', 'fire_risk_score', 'claim_filed',
                    'combined_risk', 'value_per_sqft']
    
    corr = model_data[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Features')
    plt.tight_layout()
    plt.savefig("reports/figures/correlation_heatmap.png")
    print("Saved correlation heatmap to reports/figures/correlation_heatmap.png")
    
    # Train a simple model
    print("\nTraining a simple risk prediction model...")
    X = model_data.drop('claim_filed', axis=1)
    y = model_data['claim_filed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head(5))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    print("Saved feature importance plot to reports/figures/feature_importance.png")
    
    # Load business data
    print("\nLoading business data...")
    try:
        business_data_json = pd.read_json("data/raw/sample_business_data.json")
        if "businesses" in business_data_json:
            business_data = pd.json_normalize(business_data_json["businesses"])
        else:
            business_data = business_data_json
            
        print(f"Loaded {len(business_data)} business records")
        
        # Display basic statistics
        print("\nBusiness Data Summary:")
        print(f"Industries: {business_data['industry'].value_counts().to_dict()}")
        print(f"Average revenue: ${business_data['revenue'].mean():.2f}")
        print(f"Average employees: {business_data['employees'].mean():.2f}")
        print(f"Average risk score: {business_data['risk_score'].mean():.2f}")
        
        # Analyze risk by industry
        industry_risk = business_data.groupby('industry')['risk_score'].mean().sort_values(ascending=False)
        
        print("\nAverage Risk Score by Industry:")
        for industry, risk in industry_risk.items():
            print(f"{industry}: {risk:.2f}")
        
        # Plot risk by industry
        plt.figure(figsize=(10, 6))
        sns.barplot(x=industry_risk.index, y=industry_risk.values)
        plt.title('Average Risk Score by Industry')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("reports/figures/industry_risk.png")
        print("Saved industry risk plot to reports/figures/industry_risk.png")
        
    except Exception as e:
        print(f"Error processing business data: {e}")
    
    # Business impact assessment
    print("\nPerforming business impact assessment...")
    
    # Calculate potential cost savings
    avg_manual_underwriting_time = 2.5  # hours
    avg_automated_underwriting_time = 0.5  # hours
    hourly_cost = 75  # dollars
    num_policies = 1000
    
    time_saved = (avg_manual_underwriting_time - avg_automated_underwriting_time) * num_policies
    cost_saved = time_saved * hourly_cost
    
    print(f"Efficiency Analysis:")
    print(f"Time saved with automated underwriting: {time_saved:.0f} hours")
    print(f"Cost saved: ${cost_saved:.2f}")
    
    # Calculate risk reduction impact
    baseline_loss_ratio = 0.65
    predicted_loss_ratio = 0.58
    total_premium = 10000000
    
    baseline_loss = baseline_loss_ratio * total_premium
    predicted_loss = predicted_loss_ratio * total_premium
    loss_reduction = baseline_loss - predicted_loss
    
    print(f"\nRisk Reduction Analysis:")
    print(f"Baseline expected loss: ${baseline_loss:.2f}")
    print(f"Predicted expected loss: ${predicted_loss:.2f}")
    print(f"Loss reduction: ${loss_reduction:.2f} ({(loss_reduction/baseline_loss)*100:.2f}%)")
    
    print("\nAnalysis completed successfully!")
    
except Exception as e:
    print(f"Error in data analysis: {e}")
