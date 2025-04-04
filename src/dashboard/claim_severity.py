"""
Claim Severity Prediction module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for predicting and visualizing claim severity.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout, get_property_type_colors

def generate_claim_severity_data(property_data):
    """
    Generate claim severity data based on property characteristics.
    
    Args:
        property_data (pd.DataFrame): Property data
        
    Returns:
        pd.DataFrame: Enhanced property data with claim severity predictions
    """
    # Create a copy of the data to avoid modifying the original
    data = property_data.copy()
    
    # Generate claim severity for properties with claims
    claim_properties = data[data['claim_filed'] == 1].copy()
    
    # Base severity influenced by property value (higher value = higher potential severity)
    claim_properties['base_severity'] = claim_properties['property_value'] * np.random.uniform(0.01, 0.05, len(claim_properties))
    
    # Adjust severity based on risk factors
    claim_properties['severity_multiplier'] = (
        1.0 + 
        0.2 * (claim_properties['flood_risk_score'] / 10) + 
        0.2 * (claim_properties['fire_risk_score'] / 10) + 
        0.1 * (claim_properties['property_age'] / 100)
    )
    
    # Calculate final claim severity
    claim_properties['claim_severity'] = claim_properties['base_severity'] * claim_properties['severity_multiplier']
    
    # Add severity category as a string column first, not categorical
    claim_properties['severity_category_temp'] = pd.qcut(
        claim_properties['claim_severity'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Extreme']
    )
    # Convert to string to avoid categorical issues
    claim_properties['severity_category'] = claim_properties['severity_category_temp'].astype(str)
    
    # Merge back with original data
    data = pd.merge(
        data, 
        claim_properties[['property_id', 'claim_severity', 'severity_category']], 
        on='property_id', 
        how='left'
    )
    
    # Fill NaN values for properties without claims
    data['claim_severity'] = data['claim_severity'].fillna(0)
    data['severity_category'] = data['severity_category'].fillna('No Claim')
    
    # Generate severity prediction for all properties (including those without claims)
    # This represents the expected severity if a claim were to occur
    data['predicted_severity'] = data['property_value'] * np.random.uniform(0.01, 0.05, len(data))
    data['predicted_severity'] = data['predicted_severity'] * (
        1.0 + 
        0.2 * (data['flood_risk_score'] / 10) + 
        0.2 * (data['fire_risk_score'] / 10) + 
        0.1 * (data['property_age'] / 100)
    )
    
    # Add predicted severity category as string to avoid categorical issues
    predicted_categories = pd.qcut(
        data['predicted_severity'], 
        q=4, 
        labels=['Low', 'Medium', 'High', 'Extreme']
    )
    data['predicted_severity_category'] = predicted_categories.astype(str)
    
    return data

def render_severity_heatmap(severity_data):
    """
    Render a heatmap showing claim probability vs. severity.
    
    Args:
        severity_data (pd.DataFrame): Property data with severity predictions
    """
    st.subheader("Claim Probability vs. Severity Matrix")
    
    # Create risk bins if not already present
    if 'risk_bin' not in severity_data.columns:
        severity_data['risk_bin'] = pd.qcut(
            severity_data['location_risk'], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    # Create severity bins if not already present
    if 'severity_bin' not in severity_data.columns:
        severity_data['severity_bin'] = pd.qcut(
            severity_data['predicted_severity'], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    # Calculate average claim probability for each risk-severity combination
    heatmap_data = severity_data.pivot_table(
        index='risk_bin',
        columns='severity_bin',
        values='claim_filed',
        aggfunc='mean'
    )
    
    # Calculate count for each risk-severity combination
    count_data = severity_data.pivot_table(
        index='risk_bin',
        columns='severity_bin',
        values='property_id',
        aggfunc='count'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Reds',
        text=count_data.values,
        hovertemplate='Risk: %{y}<br>Severity: %{x}<br>Claim Probability: %{z:.1%}<br>Count: %{text}<extra></extra>',
        zmin=0,
        zmax=0.3
    ))
    
    # Update layout
    custom_layout = {
        'height': 450,
        'xaxis_title': 'Predicted Claim Severity',
        'yaxis_title': 'Risk Score',
        'title': 'Claim Probability vs. Severity Matrix',
        'coloraxis_colorbar': {
            'title': 'Claim Probability',
            'tickformat': '.0%'
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This heatmap visualizes the relationship between claim probability and claim severity. 
    Darker colors indicate higher claim probabilities, while the position on the x-axis indicates the predicted severity if a claim occurs. 
    The highest risk segments are in the upper right corner, representing policies with both high claim probability and high potential severity. 
    The numbers in each cell show the count of properties in that segment.</p>
    """, unsafe_allow_html=True)

def render_severity_by_property_type(severity_data):
    """
    Render a chart showing claim severity by property type.
    
    Args:
        severity_data (pd.DataFrame): Property data with severity predictions
    """
    st.subheader("Average Claim Severity by Property Type")
    
    # Calculate average predicted severity by property type
    severity_by_type = severity_data.groupby('property_type')['predicted_severity'].mean().reset_index()
    
    # Calculate actual severity for properties with claims
    actual_severity = severity_data[severity_data['claim_filed'] == 1].groupby('property_type')['claim_severity'].mean().reset_index()
    actual_severity.columns = ['property_type', 'actual_severity']
    
    # Merge the data
    severity_comparison = pd.merge(severity_by_type, actual_severity, on='property_type', how='left')
    severity_comparison['actual_severity'] = severity_comparison['actual_severity'].fillna(0)
    
    # Create bar chart
    fig = go.Figure()
    
    # Add predicted severity bars
    fig.add_trace(go.Bar(
        x=severity_comparison['property_type'],
        y=severity_comparison['predicted_severity'],
        name='Predicted Severity',
        marker_color=COLORS['primary']
    ))
    
    # Add actual severity bars for properties with claims
    fig.add_trace(go.Bar(
        x=severity_comparison['property_type'],
        y=severity_comparison['actual_severity'],
        name='Actual Severity (Claims Only)',
        marker_color=COLORS['secondary']
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Property Type',
        'yaxis_title': 'Average Claim Severity ($)',
        'barmode': 'group',
        'title': 'Predicted vs. Actual Claim Severity by Property Type',
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
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares the predicted claim severity with the actual severity for properties that have filed claims, broken down by property type. 
    Industrial properties typically have the highest severity due to their higher property values and complex risks. 
    The difference between predicted and actual severity can help calibrate the severity prediction model.</p>
    """, unsafe_allow_html=True)

def render_severity_factors(severity_data):
    """
    Render a chart showing the impact of different factors on claim severity.
    
    Args:
        severity_data (pd.DataFrame): Property data with severity predictions
    """
    st.subheader("Factors Influencing Claim Severity")
    
    # Calculate correlation between various factors and predicted severity
    corr_data = severity_data[['predicted_severity', 'property_value', 'property_age', 'property_size', 'flood_risk_score', 'fire_risk_score', 'location_risk']]
    correlations = corr_data.corr()['predicted_severity'].drop('predicted_severity').sort_values(ascending=False)
    
    # Create bar chart of correlations
    fig = go.Figure()
    
    # Add correlation bars
    fig.add_trace(go.Bar(
        x=correlations.index,
        y=correlations.values,
        marker_color=[COLORS['primary'] if v > 0 else COLORS['accent'] for v in correlations.values]
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Factor',
        'yaxis_title': 'Correlation with Claim Severity',
        'yaxis_range': [-1, 1],
        'title': 'Factors Influencing Claim Severity'
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart shows the correlation between various property characteristics and claim severity. 
    Factors with higher positive correlation have a stronger influence on increasing claim severity. 
    Property value typically has the strongest correlation with severity, as more valuable properties naturally have higher potential losses. 
    Understanding these relationships helps underwriters identify which factors to focus on when assessing potential claim costs.</p>
    """, unsafe_allow_html=True)

def render_severity_distribution(severity_data):
    """
    Render a chart showing the distribution of predicted claim severity.
    
    Args:
        severity_data (pd.DataFrame): Property data with severity predictions
    """
    st.subheader("Claim Severity Distribution")
    
    # Create histogram of predicted severity
    fig = px.histogram(
        severity_data,
        x='predicted_severity',
        color='property_type',
        nbins=50,
        opacity=0.7,
        color_discrete_map=get_property_type_colors()
    )
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Predicted Claim Severity ($)',
        'yaxis_title': 'Count',
        'bargap': 0.1,
        'title': 'Distribution of Predicted Claim Severity by Property Type',
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
    <p style='margin-top: 10px; margin-bottom: 20px;'>This histogram shows the distribution of predicted claim severity across the portfolio. 
    The distribution is typically right-skewed, with many properties having relatively low predicted severity and fewer properties with very high severity. 
    Different property types show distinct severity profiles, with industrial and commercial properties skewing toward higher severity values.</p>
    """, unsafe_allow_html=True)

def render_claim_severity_page(property_data):
    """Render the Claim Severity Prediction page with interactive visualizations."""
    st.markdown('<h2 class="section-header">Claim Severity Prediction</h2>', unsafe_allow_html=True)
    
    # Generate claim severity data
    severity_data = generate_claim_severity_data(property_data)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page provides predictions and analysis of potential claim severity, 
    helping underwriters assess not just the likelihood of claims but also their potential financial impact.</p>
    """, unsafe_allow_html=True)
    
    # Render severity heatmap
    render_severity_heatmap(severity_data)
    
    # Create columns for the next two visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Render severity by property type
        render_severity_by_property_type(severity_data)
    
    with col2:
        # Render severity factors
        render_severity_factors(severity_data)
    
    # Render severity distribution
    render_severity_distribution(severity_data)
