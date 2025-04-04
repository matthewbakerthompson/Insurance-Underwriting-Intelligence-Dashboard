"""
Data loading module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for loading and preprocessing data.
"""
import pandas as pd
import numpy as np
import os
import json

def load_sample_data():
    """
    Generate sample property and business data for the dashboard.
    
    Returns:
        tuple: (property_data, business_data, model_performance)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample property data
    num_properties = 1253
    property_types = ['Residential', 'Commercial', 'Industrial']
    property_type_probs = [0.7, 0.2, 0.1]  # 70% residential, 20% commercial, 10% industrial
    
    property_data = pd.DataFrame({
        'property_id': [f'PROP-{i:05d}' for i in range(1, num_properties + 1)],
        'property_type': np.random.choice(property_types, size=num_properties, p=property_type_probs),
        'property_value': np.random.normal(350000, 150000, num_properties).astype(int),
        'property_age': np.random.gamma(shape=5, scale=5, size=num_properties).astype(int) + 1,
        'property_size': np.random.normal(2500, 1000, num_properties).astype(int),
    })
    
    # Adjust values based on property type - ensure proper type conversion
    property_data.loc[property_data['property_type'] == 'Commercial', 'property_value'] = (property_data.loc[property_data['property_type'] == 'Commercial', 'property_value'] * 2.5).astype(int)
    property_data.loc[property_data['property_type'] == 'Commercial', 'property_size'] = (property_data.loc[property_data['property_type'] == 'Commercial', 'property_size'] * 3).astype(int)
    property_data.loc[property_data['property_type'] == 'Industrial', 'property_value'] = (property_data.loc[property_data['property_type'] == 'Industrial', 'property_value'] * 3.5).astype(int)
    property_data.loc[property_data['property_type'] == 'Industrial', 'property_size'] = (property_data.loc[property_data['property_type'] == 'Industrial', 'property_size'] * 5).astype(int)
    
    # Ensure all values are positive
    property_data['property_value'] = property_data['property_value'].clip(lower=50000)
    property_data['property_size'] = property_data['property_size'].clip(lower=500)
    property_data['property_age'] = property_data['property_age'].clip(lower=1)
    
    # Generate risk scores with some correlation to property characteristics
    property_data['flood_risk_score'] = (
        0.2 * (property_data['property_age'] / 50) + 
        0.3 * np.random.normal(5, 2, num_properties)
    ).clip(0, 10)
    
    property_data['fire_risk_score'] = (
        0.3 * (property_data['property_age'] / 50) + 
        0.2 * (property_data['property_size'] / 5000) +
        0.5 * np.random.normal(5, 2, num_properties)
    ).clip(0, 10)
    
    # Location risk (combined risk)
    property_data['location_risk'] = (
        0.4 * property_data['flood_risk_score'] + 
        0.4 * property_data['fire_risk_score'] + 
        0.2 * np.random.normal(5, 1, num_properties)
    ).clip(0, 10)
    
    # Generate claim data with correlation to risk scores
    claim_prob = 1 / (1 + np.exp(-0.5 * (property_data['location_risk'] - 5)))
    property_data['claim_filed'] = np.random.binomial(1, claim_prob)
    
    # Create bins for analysis
    property_data['value_bin'] = pd.qcut(
        property_data['property_value'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    property_data['flood_risk_bin'] = pd.qcut(
        property_data['flood_risk_score'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    property_data['fire_risk_bin'] = pd.qcut(
        property_data['fire_risk_score'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Generate sample business data
    num_businesses = 528
    industries = [
        'Retail', 'Manufacturing', 'Healthcare', 'Technology', 
        'Finance', 'Construction', 'Food Service', 'Transportation'
    ]
    
    business_data = pd.DataFrame({
        'business_id': [f'BUS-{i:05d}' for i in range(1, num_businesses + 1)],
        'industry': np.random.choice(industries, size=num_businesses),
        'annual_revenue': np.random.lognormal(mean=12, sigma=1, size=num_businesses).astype(int),
        'employee_count': np.random.lognormal(mean=3, sigma=1.2, size=num_businesses).astype(int),
        'years_in_operation': np.random.gamma(shape=2, scale=10, size=num_businesses).astype(int) + 1,
    })
    
    # Simulate model performance data
    model_performance = {
        'metrics': {
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.75,
            'f1': 0.77,
            'roc_auc': 0.85
        },
        'feature_importance': pd.DataFrame({
            'Feature': [
                'Flood Risk Score', 
                'Property Age', 
                'Fire Risk Score', 
                'Property Value', 
                'Property Size', 
                'Property Type'
            ],
            'Importance': [0.28, 0.22, 0.18, 0.15, 0.12, 0.05]
        })
    }
    
    return property_data, business_data, model_performance
