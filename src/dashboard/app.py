"""
Insurance Underwriting Dashboard

A Streamlit dashboard for visualizing insurance underwriting data,
model performance, and risk insights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modular components
from theme import apply_theme, COLORS, PLOT_LAYOUT
from load_data import load_sample_data
from overview import render_overview_page
from risk_analysis import render_risk_analysis_page
from claims_analysis import render_claims_analysis_page
from model_performance import render_model_performance_page
from what_if_analysis import render_what_if_analysis_page

# Import new modules
from time_series_analysis import render_time_series_page
from geographic_analysis import render_geographic_analysis_page
from portfolio_diversification import render_portfolio_diversification_page
from claim_severity import render_claim_severity_page
from comparative_analysis import render_comparative_analysis_page
from risk_correlation import render_risk_correlation_page
from scenario_planning import render_scenario_planning_page
from underwriting_recommendations import render_underwriting_recommendations_page
from regulatory_compliance import render_regulatory_compliance_page

# Set page configuration
st.set_page_config(
    page_title="Insurance Underwriting Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
apply_theme()

# Dashboard title and description
st.markdown('<h1 class="main-header">Insurance Underwriting Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Insights into insurance underwriting data, risk factors, and predictive model performance in a dynamic visualization dashboard.</p>', unsafe_allow_html=True)

# Load sample data
property_data, business_data, model_performance = load_sample_data()

# Sidebar navigation
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    [
        "Overview", 
        "Risk Analysis", 
        "Claims Analysis", 
        "Model Performance", 
        "What-If Analysis",
        "Time Series Analysis",
        "Geographic Risk Visualization",
        "Portfolio Diversification",
        "Claim Severity Prediction",
        "Comparative Analysis",
        "Risk Correlation Matrix",
        "Scenario Planning",
        "Underwriting Recommendations",
        "Regulatory Compliance"
    ],
    label_visibility="collapsed"
)

# Render the selected page
if page == "Overview":
    render_overview_page(property_data, business_data)
    
elif page == "Risk Analysis":
    render_risk_analysis_page(property_data)
    
elif page == "Claims Analysis":
    render_claims_analysis_page(property_data)
    
elif page == "Model Performance":
    render_model_performance_page(model_performance)
    
elif page == "What-If Analysis":
    render_what_if_analysis_page(property_data)
    
# New pages
elif page == "Time Series Analysis":
    render_time_series_page(property_data)
    
elif page == "Geographic Risk Visualization":
    render_geographic_analysis_page(property_data)
    
elif page == "Portfolio Diversification":
    render_portfolio_diversification_page(property_data, business_data)
    
elif page == "Claim Severity Prediction":
    render_claim_severity_page(property_data)
    
elif page == "Comparative Analysis":
    render_comparative_analysis_page(property_data)
    
elif page == "Risk Correlation Matrix":
    render_risk_correlation_page(property_data)
    
elif page == "Scenario Planning":
    render_scenario_planning_page()
    
elif page == "Underwriting Recommendations":
    render_underwriting_recommendations_page(property_data)
    
elif page == "Regulatory Compliance":
    render_regulatory_compliance_page()

# Add footer
st.markdown("---")
st.markdown("<div style='color: #64748B; font-size: 0.8rem;'>Insurance Underwriting Intelligence Project by Matthew Thompson</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # This block will be executed when the script is run directly
    pass