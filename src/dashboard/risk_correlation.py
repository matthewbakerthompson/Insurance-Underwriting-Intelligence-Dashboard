"""
Risk Factor Correlation Matrix module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for analyzing correlations between risk factors.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def render_correlation_matrix(property_data):
    """
    Render a correlation matrix of risk factors.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Risk Factor Correlation Matrix")
    
    # Select relevant columns for correlation analysis
    corr_columns = [
        'property_value', 'property_age', 'property_size', 
        'flood_risk_score', 'fire_risk_score', 'location_risk', 'claim_filed'
    ]
    
    # Create correlation matrix
    corr_matrix = property_data[corr_columns].corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    # Update layout
    custom_layout = {
        'height': 600,
        'title': 'Risk Factor Correlation Matrix',
        'xaxis': {
            'title': None,
            'tickangle': 45
        },
        'yaxis': {
            'title': None
        },
        'coloraxis_colorbar': {
            'title': 'Correlation'
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This correlation matrix shows the relationships between different risk factors and property characteristics. 
    Values closer to 1 indicate strong positive correlation, values closer to -1 indicate strong negative correlation, and values near 0 indicate little to no correlation. 
    Understanding these relationships helps identify which factors tend to occur together, potentially compounding risk.</p>
    """, unsafe_allow_html=True)

def render_key_correlations(property_data):
    """
    Render a bar chart of key correlations with claim filing.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Key Correlations with Claims")
    
    # Select relevant columns for correlation analysis
    corr_columns = [
        'property_value', 'property_age', 'property_size', 
        'flood_risk_score', 'fire_risk_score', 'location_risk'
    ]
    
    # Calculate correlations with claim_filed
    correlations = []
    for col in corr_columns:
        corr = property_data[col].corr(property_data['claim_filed'])
        correlations.append({
            'Factor': col.replace('_', ' ').title(),
            'Correlation': corr
        })
    
    # Create dataframe and sort by absolute correlation
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=corr_df['Factor'],
        y=corr_df['Correlation'],
        marker_color=[COLORS['primary'] if v > 0 else COLORS['manufacturing'] for v in corr_df['Correlation']]  # Using manufacturing (pink) instead of non-existent 'red'
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Risk Factor',
        'yaxis_title': 'Correlation with Claim Filing',
        'yaxis_range': [-1, 1],
        'title': 'Correlation Between Risk Factors and Claims'
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart highlights the correlation between various factors and claim filing. 
    Factors with higher positive correlation are more strongly associated with claims being filed. 
    This information helps underwriters prioritize which risk factors to focus on during the underwriting process.</p>
    """, unsafe_allow_html=True)

def render_scatter_matrix(property_data):
    """
    Render a scatter matrix of key risk factors.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Risk Factor Relationships")
    
    # Select relevant columns and sample data for performance
    scatter_columns = [
        'property_age', 'flood_risk_score', 'fire_risk_score', 'location_risk'
    ]
    
    # Sample data for better performance (adjust sample size as needed)
    sample_data = property_data.sample(min(500, len(property_data)))
    
    # Create scatter matrix
    fig = px.scatter_matrix(
        sample_data,
        dimensions=scatter_columns,
        color='claim_filed',
        color_discrete_map={0: COLORS['primary'], 1: COLORS['manufacturing']},  # Using primary instead of 'blue' and manufacturing (pink) instead of 'red'
        title="Relationships Between Risk Factors",
        labels={
            'property_age': 'Property Age',
            'flood_risk_score': 'Flood Risk',
            'fire_risk_score': 'Fire Risk',
            'location_risk': 'Overall Risk',
            'claim_filed': 'Claim Filed'
        }
    )
    
    # Update layout
    custom_layout = {
        'height': 700,
        'title': {'text': 'Scatter Matrix of Risk Factors', 'font': {'size': 20, 'color': '#0F172A'}}
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This scatter matrix shows the relationships between pairs of risk factors, with points colored by claim status. 
    Clusters of red points (claims) in certain regions indicate combinations of factors that are associated with higher claim likelihood. 
    This visualization helps identify non-linear relationships and interaction effects between risk factors that might not be apparent in the correlation matrix.</p>
    """, unsafe_allow_html=True)

def render_compound_risk_analysis(property_data):
    """
    Render analysis of compound risk situations.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Compound Risk Analysis")
    
    # Create high risk indicators
    property_data['high_flood_risk'] = property_data['flood_risk_score'] > 7
    property_data['high_fire_risk'] = property_data['fire_risk_score'] > 7
    property_data['high_age'] = property_data['property_age'] > 30
    
    # Count properties with different risk combinations
    risk_combinations = [
        {'name': 'No High Risks', 'condition': ~(property_data['high_flood_risk'] | property_data['high_fire_risk'] | property_data['high_age'])},
        {'name': 'High Flood Risk Only', 'condition': property_data['high_flood_risk'] & ~property_data['high_fire_risk'] & ~property_data['high_age']},
        {'name': 'High Fire Risk Only', 'condition': ~property_data['high_flood_risk'] & property_data['high_fire_risk'] & ~property_data['high_age']},
        {'name': 'High Age Only', 'condition': ~property_data['high_flood_risk'] & ~property_data['high_fire_risk'] & property_data['high_age']},
        {'name': 'High Flood & Fire Risk', 'condition': property_data['high_flood_risk'] & property_data['high_fire_risk'] & ~property_data['high_age']},
        {'name': 'High Flood Risk & Age', 'condition': property_data['high_flood_risk'] & ~property_data['high_fire_risk'] & property_data['high_age']},
        {'name': 'High Fire Risk & Age', 'condition': ~property_data['high_flood_risk'] & property_data['high_fire_risk'] & property_data['high_age']},
        {'name': 'All High Risk Factors', 'condition': property_data['high_flood_risk'] & property_data['high_fire_risk'] & property_data['high_age']}
    ]
    
    # Calculate counts and claim rates for each combination
    compound_risk_data = []
    for combo in risk_combinations:
        subset = property_data[combo['condition']]
        count = len(subset)
        if count > 0:
            claim_rate = subset['claim_filed'].mean()
            avg_risk = subset['location_risk'].mean()
        else:
            claim_rate = 0
            avg_risk = 0
            
        compound_risk_data.append({
            'Risk Combination': combo['name'],
            'Count': count,
            'Claim Rate': claim_rate,
            'Avg Risk Score': avg_risk
        })
    
    # Create dataframe
    compound_df = pd.DataFrame(compound_risk_data)
    
    # Create bar chart
    fig = go.Figure()
    
    # Add claim rate bars
    fig.add_trace(go.Bar(
        x=compound_df['Risk Combination'],
        y=compound_df['Claim Rate'],
        name='Claim Rate',
        marker_color=COLORS['primary'],
        yaxis='y'
    ))
    
    # Add average risk score line
    fig.add_trace(go.Scatter(
        x=compound_df['Risk Combination'],
        y=compound_df['Avg Risk Score'],
        name='Avg Risk Score',
        mode='lines+markers',
        marker=dict(color=COLORS['finance']),  # Using finance instead of non-existent 'orange'
        yaxis='y2'
    ))
    
    # Update layout
    custom_layout = {
        'height': 500,
        'title': 'Compound Risk Factors and Claim Rates',
        'xaxis_title': 'Risk Combination',
        'yaxis': {
            'title': 'Claim Rate',
            'tickformat': '.0%',
            'side': 'left',
            'range': [0, compound_df['Claim Rate'].max() * 1.2]
        },
        'yaxis2': {
            'title': 'Avg Risk Score',
            'overlaying': 'y',
            'side': 'right',
            'range': [0, 10]
        },
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        'barmode': 'group'
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    compound_df['Claim Rate'] = compound_df['Claim Rate'].apply(lambda x: f"{x:.1%}")
    compound_df['Avg Risk Score'] = compound_df['Avg Risk Score'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        compound_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This analysis examines how combinations of high-risk factors affect claim rates and overall risk scores. 
    Properties with multiple high-risk factors typically show significantly higher claim rates than those with only one high-risk factor or none. 
    This demonstrates the compound effect of risk factors, where the combined impact is greater than the sum of individual risks. 
    Underwriters should pay special attention to properties with multiple high-risk indicators.</p>
    """, unsafe_allow_html=True)

def render_risk_correlation_page(property_data):
    """Render the Risk Factor Correlation Matrix page with interactive visualizations."""
    st.markdown('<h2 class="section-header">Risk Factor Correlation Matrix</h2>', unsafe_allow_html=True)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page analyzes the relationships between different risk factors, 
    helping underwriters understand how risks interact and identify compound risk situations.</p>
    """, unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Risk Relationships", "Compound Risk"])
    
    with tab1:
        # Render correlation matrix
        render_correlation_matrix(property_data)
        
        # Render key correlations
        render_key_correlations(property_data)
    
    with tab2:
        # Render scatter matrix
        render_scatter_matrix(property_data)
    
    with tab3:
        # Render compound risk analysis
        render_compound_risk_analysis(property_data)
