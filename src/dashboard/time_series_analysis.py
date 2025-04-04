"""
Time Series Analysis module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating time series visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def generate_time_series_data(property_data):
    """
    Generate time series data for risk scores and claim rates.
    
    Args:
        property_data (pd.DataFrame): Property data
        
    Returns:
        pd.DataFrame: Time series data with dates and metrics
    """
    # Create a date range for the past 2 years (monthly data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # Approximately 2 years
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Create time series data frame
    time_series_data = pd.DataFrame({
        'date': date_range
    })
    
    # Generate risk scores over time with seasonal patterns
    # Base trend with slight increase over time
    base_trend = np.linspace(4.8, 5.2, len(date_range))
    
    # Add seasonal component (higher in summer months)
    month_effect = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.4, 0.2, 0.0, -0.1])
    seasonal_component = np.array([month_effect[d.month-1] for d in date_range])
    
    # Add random noise
    noise = np.random.normal(0, 0.15, len(date_range))
    
    # Combine components for average risk score
    time_series_data['avg_risk_score'] = base_trend + seasonal_component + noise
    time_series_data['avg_risk_score'] = time_series_data['avg_risk_score'].clip(0, 10)
    
    # Generate claim rates with correlation to risk scores
    # Base claim rate with seasonal pattern
    base_claim_rate = 0.08 + 0.04 * seasonal_component
    
    # Add correlation with risk score and some noise
    claim_rate_noise = np.random.normal(0, 0.01, len(date_range))
    time_series_data['claim_rate'] = base_claim_rate + 0.01 * (time_series_data['avg_risk_score'] - 5) + claim_rate_noise
    time_series_data['claim_rate'] = time_series_data['claim_rate'].clip(0, 1)
    
    # Generate property type specific risk scores
    for prop_type in ['Residential', 'Commercial', 'Industrial']:
        # Different base levels for different property types
        if prop_type == 'Residential':
            base = 4.5
            volatility = 0.12
        elif prop_type == 'Commercial':
            base = 5.2
            volatility = 0.18
        else:  # Industrial
            base = 5.8
            volatility = 0.25
            
        # Generate the time series with the same seasonal pattern but different base and volatility
        type_noise = np.random.normal(0, volatility, len(date_range))
        time_series_data[f'{prop_type.lower()}_risk_score'] = base + base_trend - 5 + seasonal_component * 1.2 + type_noise
        time_series_data[f'{prop_type.lower()}_risk_score'] = time_series_data[f'{prop_type.lower()}_risk_score'].clip(0, 10)
        
        # Generate claim rates for each property type
        type_claim_noise = np.random.normal(0, 0.015, len(date_range))
        time_series_data[f'{prop_type.lower()}_claim_rate'] = (
            base_claim_rate + 
            0.01 * (time_series_data[f'{prop_type.lower()}_risk_score'] - 5) + 
            type_claim_noise
        )
        time_series_data[f'{prop_type.lower()}_claim_rate'] = time_series_data[f'{prop_type.lower()}_claim_rate'].clip(0, 1)
    
    return time_series_data

def render_risk_score_time_series(time_series_data):
    """
    Render a time series chart of risk scores.
    
    Args:
        time_series_data (pd.DataFrame): Time series data
    """
    st.subheader("Risk Score Trends Over Time")
    
    # Create plot
    fig = go.Figure()
    
    # Add overall average risk score
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['avg_risk_score'],
        mode='lines',
        name='Overall Average',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    # Add property type specific risk scores
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['residential_risk_score'],
        mode='lines',
        name='Residential',
        line=dict(color=COLORS['residential'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['commercial_risk_score'],
        mode='lines',
        name='Commercial',
        line=dict(color=COLORS['commercial'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['industrial_risk_score'],
        mode='lines',
        name='Industrial',
        line=dict(color=COLORS['industrial'], width=2)
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'title': 'Risk Score Trends by Property Type',
        'xaxis_title': 'Date',
        'yaxis_title': 'Risk Score (0-10)',
        'yaxis_range': [0, 10],
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart shows how risk scores have changed over time for different property types. 
    The seasonal patterns indicate higher risk during summer months, likely due to increased weather-related events. 
    Industrial properties consistently show the highest risk, followed by commercial and residential properties.</p>
    """, unsafe_allow_html=True)

def render_claim_rate_time_series(time_series_data):
    """
    Render a time series chart of claim rates.
    
    Args:
        time_series_data (pd.DataFrame): Time series data
    """
    st.subheader("Claim Rate Trends Over Time")
    
    # Create plot
    fig = go.Figure()
    
    # Add overall average claim rate
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['claim_rate'],
        mode='lines',
        name='Overall Average',
        line=dict(color=COLORS['primary'], width=3)
    ))
    
    # Add property type specific claim rates
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['residential_claim_rate'],
        mode='lines',
        name='Residential',
        line=dict(color=COLORS['residential'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['commercial_claim_rate'],
        mode='lines',
        name='Commercial',
        line=dict(color=COLORS['commercial'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_series_data['date'],
        y=time_series_data['industrial_claim_rate'],
        mode='lines',
        name='Industrial',
        line=dict(color=COLORS['industrial'], width=2)
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'title': 'Claim Rate Trends by Property Type',
        'xaxis_title': 'Date',
        'yaxis_title': 'Claim Rate',
        'yaxis_range': [0, 0.2],
        'yaxis_tickformat': '.1%',
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart displays claim rates over time for different property types. 
    Claim rates closely follow risk score patterns, with seasonal increases during summer months. 
    Industrial properties show the highest claim rates, which aligns with their higher risk profiles. 
    This information can help underwriters anticipate claim volume and adjust pricing strategies accordingly.</p>
    """, unsafe_allow_html=True)

def render_time_series_page(property_data):
    """Render the Time Series Analysis page with interactive visualizations."""
    st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)
    
    # Generate time series data
    time_series_data = generate_time_series_data(property_data)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page provides temporal analysis of risk scores and claim rates, 
    helping underwriters identify trends, seasonal patterns, and long-term changes in risk profiles.</p>
    """, unsafe_allow_html=True)
    
    # Create columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Render risk score time series
        render_risk_score_time_series(time_series_data)
    
    with col2:
        # Render claim rate time series
        render_claim_rate_time_series(time_series_data)
    
    # Add year-over-year comparison
    st.subheader("Year-over-Year Comparison")
    
    # Calculate YoY changes
    current_year = time_series_data.iloc[-12:].reset_index(drop=True)
    previous_year = time_series_data.iloc[-24:-12].reset_index(drop=True)
    
    # Calculate average metrics for each year
    current_avg_risk = current_year['avg_risk_score'].mean()
    previous_avg_risk = previous_year['avg_risk_score'].mean()
    risk_change_pct = (current_avg_risk - previous_avg_risk) / previous_avg_risk * 100
    
    current_avg_claim = current_year['claim_rate'].mean()
    previous_avg_claim = previous_year['claim_rate'].mean()
    claim_change_pct = (current_avg_claim - previous_avg_claim) / previous_avg_claim * 100
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Average Risk Score (Current Year)", 
            value=f"{current_avg_risk:.2f}",
            delta=f"{risk_change_pct:.1f}% vs Previous Year"
        )
    
    with col2:
        st.metric(
            label="Average Claim Rate (Current Year)", 
            value=f"{current_avg_claim:.1%}",
            delta=f"{claim_change_pct:.1f}% vs Previous Year"
        )
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>The year-over-year comparison shows how risk scores and claim rates 
    have changed compared to the previous year. This helps identify long-term trends and assess whether risk is increasing or decreasing over time.</p>
    """, unsafe_allow_html=True)
