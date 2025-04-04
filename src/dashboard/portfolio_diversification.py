"""
Portfolio Diversification Analysis module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for analyzing portfolio diversification and concentration risks.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout, get_property_type_colors

def render_portfolio_composition(property_data, business_data):
    """
    Render a treemap of portfolio composition by property type and industry.
    
    Args:
        property_data (pd.DataFrame): Property data
        business_data (pd.DataFrame): Business data
    """
    st.subheader("Portfolio Composition")
    
    # Create property type and value data
    property_value_by_type = property_data.groupby('property_type')['property_value'].sum().reset_index()
    property_value_by_type['category'] = 'Property Type'
    property_value_by_type.columns = ['subcategory', 'value', 'category']
    
    # Create industry and revenue data
    industry_revenue = business_data.groupby('industry')['annual_revenue'].sum().reset_index()
    industry_revenue['category'] = 'Industry'
    industry_revenue.columns = ['subcategory', 'value', 'category']
    
    # Combine the data
    portfolio_data = pd.concat([property_value_by_type, industry_revenue], ignore_index=True)
    
    # Create treemap
    fig = px.treemap(
        portfolio_data,
        path=['category', 'subcategory'],
        values='value',
        color='category',
        color_discrete_map={
            'Property Type': COLORS['primary'],
            'Industry': COLORS['secondary']
        }
    )
    
    # Update layout
    custom_layout = {
        'height': 500,
        'margin': {"r": 0, "t": 0, "l": 0, "b": 0}
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    fig.update_traces(textinfo="label+percent entry")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This treemap visualizes the composition of your insurance portfolio by property type and industry sector. 
    The size of each block represents the total value (for properties) or revenue (for businesses) in that category. 
    This visualization helps identify concentration risks where your portfolio may be overexposed to specific sectors or property types.</p>
    """, unsafe_allow_html=True)

def render_risk_concentration(property_data):
    """
    Render a heatmap of risk concentration by property type and value.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Risk Concentration Analysis")
    
    # Create value bins if not already present
    if 'value_bin' not in property_data.columns:
        property_data['value_bin'] = pd.qcut(
            property_data['property_value'], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    # Create a pivot table of average risk score by property type and value bin
    risk_pivot = property_data.pivot_table(
        index='property_type',
        columns='value_bin',
        values='location_risk',
        aggfunc='mean'
    ).round(2)
    
    # Create a pivot table of property count by property type and value bin
    count_pivot = property_data.pivot_table(
        index='property_type',
        columns='value_bin',
        values='property_id',
        aggfunc='count'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=risk_pivot.values,
        x=risk_pivot.columns,
        y=risk_pivot.index,
        colorscale='Reds',
        text=count_pivot.values,
        hovertemplate='Property Type: %{y}<br>Value: %{x}<br>Avg Risk Score: %{z}<br>Count: %{text}<extra></extra>',
        zmin=3,
        zmax=7
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Property Value',
        'yaxis_title': 'Property Type',
        'title': 'Risk Concentration by Property Type and Value',
        'coloraxis_colorbar': {
            'title': 'Risk Score'
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This heatmap shows the concentration of risk across different property types and value segments. 
    Darker colors indicate higher average risk scores, while the numbers represent the count of properties in each segment. 
    This visualization helps identify high-risk segments within your portfolio that may require additional attention or risk mitigation strategies.</p>
    """, unsafe_allow_html=True)

def render_diversification_metrics(property_data, business_data):
    """
    Render diversification metrics and indicators.
    
    Args:
        property_data (pd.DataFrame): Property data
        business_data (pd.DataFrame): Business data
    """
    st.subheader("Diversification Metrics")
    
    # Calculate property type concentration (Herfindahl-Hirschman Index)
    property_type_counts = property_data['property_type'].value_counts()
    property_type_percentages = property_type_counts / property_type_counts.sum()
    property_hhi = (property_type_percentages ** 2).sum()
    
    # Calculate industry concentration
    industry_counts = business_data['industry'].value_counts()
    industry_percentages = industry_counts / industry_counts.sum()
    industry_hhi = (industry_percentages ** 2).sum()
    
    # Calculate risk score distribution
    risk_std = property_data['location_risk'].std()
    risk_range = property_data['location_risk'].max() - property_data['location_risk'].min()
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Property Type Diversification
        diversification_score = 1 - property_hhi
        st.metric(
            label="Property Type Diversification", 
            value=f"{diversification_score:.2f}",
            help="Score from 0 to 1, where higher values indicate better diversification across property types."
        )
        
        # Show property type distribution
        property_dist = pd.DataFrame({
            'Property Type': property_type_counts.index,
            'Count': property_type_counts.values,
            'Percentage': (property_type_percentages * 100).values
        })
        
        st.dataframe(property_dist, hide_index=True)
    
    with col2:
        # Industry Diversification
        industry_diversification = 1 - industry_hhi
        st.metric(
            label="Industry Diversification", 
            value=f"{industry_diversification:.2f}",
            help="Score from 0 to 1, where higher values indicate better diversification across industry sectors."
        )
        
        # Show industry distribution
        industry_dist = pd.DataFrame({
            'Industry': industry_counts.index,
            'Count': industry_counts.values,
            'Percentage': (industry_percentages * 100).values
        })
        
        st.dataframe(industry_dist, hide_index=True)
    
    with col3:
        # Risk Diversification
        risk_diversification = 1 - (risk_std / risk_range)
        st.metric(
            label="Risk Diversification", 
            value=f"{risk_diversification:.2f}",
            help="Score from 0 to 1, where higher values indicate better diversification of risk levels."
        )
        
        # Show risk distribution stats
        st.write("Risk Score Statistics:")
        risk_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                f"{property_data['location_risk'].mean():.2f}",
                f"{property_data['location_risk'].median():.2f}",
                f"{risk_std:.2f}",
                f"{property_data['location_risk'].min():.2f}",
                f"{property_data['location_risk'].max():.2f}"
            ]
        })
        
        st.dataframe(risk_stats, hide_index=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>These metrics quantify the level of diversification in your insurance portfolio. 
    Higher diversification scores indicate a more balanced portfolio with reduced concentration risk. 
    The Property Type and Industry Diversification scores measure how evenly your exposure is spread across different categories, 
    while the Risk Diversification score measures the distribution of risk levels within your portfolio.</p>
    """, unsafe_allow_html=True)

def render_optimal_allocation(property_data):
    """
    Render a chart showing current vs. optimal portfolio allocation.
    
    Args:
        property_data (pd.DataFrame): Property data
    """
    st.subheader("Current vs. Optimal Portfolio Allocation")
    
    # Calculate current allocation
    current_allocation = property_data.groupby('property_type')['property_value'].sum()
    current_percentage = current_allocation / current_allocation.sum()
    
    # Define "optimal" allocation based on risk-adjusted returns
    # This is a simplified model - in reality, this would be based on more complex calculations
    property_risk = property_data.groupby('property_type')['location_risk'].mean()
    claim_rates = property_data.groupby('property_type')['claim_filed'].mean()
    
    # Calculate a simple risk-adjusted return metric
    # Lower risk and lower claim rates are better
    risk_adjusted_metric = 1 / (property_risk * claim_rates)
    optimal_percentage = risk_adjusted_metric / risk_adjusted_metric.sum()
    
    # Create comparison dataframe
    allocation_comparison = pd.DataFrame({
        'Property Type': current_percentage.index,
        'Current Allocation': current_percentage.values * 100,
        'Optimal Allocation': optimal_percentage.values * 100,
        'Difference': (optimal_percentage.values - current_percentage.values) * 100
    })
    
    # Create bar chart
    fig = go.Figure()
    
    # Add current allocation bars
    fig.add_trace(go.Bar(
        x=allocation_comparison['Property Type'],
        y=allocation_comparison['Current Allocation'],
        name='Current Allocation',
        marker_color=COLORS['primary']
    ))
    
    # Add optimal allocation bars
    fig.add_trace(go.Bar(
        x=allocation_comparison['Property Type'],
        y=allocation_comparison['Optimal Allocation'],
        name='Optimal Allocation',
        marker_color=COLORS['secondary']
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Property Type',
        'yaxis_title': 'Portfolio Allocation (%)',
        'barmode': 'group',
        'title': 'Current vs. Risk-Optimized Portfolio Allocation',
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
    
    # Show allocation comparison table
    st.dataframe(allocation_comparison.round(1), hide_index=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares your current portfolio allocation with an optimal allocation based on risk-adjusted metrics. 
    The optimal allocation aims to maximize returns while minimizing risk exposure. 
    Positive differences suggest increasing allocation to that property type, while negative differences suggest reducing exposure. 
    This analysis can guide strategic decisions about which segments to grow or reduce in your portfolio.</p>
    """, unsafe_allow_html=True)

def render_portfolio_diversification_page(property_data, business_data):
    """Render the Portfolio Diversification Analysis page with interactive visualizations."""
    st.markdown('<h2 class="section-header">Portfolio Diversification Analysis</h2>', unsafe_allow_html=True)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page analyzes the diversification of your insurance portfolio across property types, 
    industries, and risk levels to identify concentration risks and opportunities for better risk distribution.</p>
    """, unsafe_allow_html=True)
    
    # Render portfolio composition treemap
    render_portfolio_composition(property_data, business_data)
    
    # Create columns for the next two visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Render risk concentration heatmap
        render_risk_concentration(property_data)
    
    with col2:
        # Render optimal allocation chart
        render_optimal_allocation(property_data)
    
    # Render diversification metrics
    render_diversification_metrics(property_data, business_data)
