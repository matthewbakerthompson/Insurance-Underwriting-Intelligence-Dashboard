"""
Risk Analysis module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating risk analysis visualizations.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from theme import COLORS, PLOT_LAYOUT, styled_container, get_property_type_colors, merge_layout

def render_correlation_heatmap(property_data):
    """Render a correlation heatmap of key risk factors."""
    st.markdown('<h4 class="section-header">Key Risk Factors Correlation Analysis</h4>', unsafe_allow_html=True)
    
    def plot_correlation():
        # Select only numeric columns for correlation
        numeric_cols = property_data.select_dtypes(include=[np.number]).columns.tolist()
        # Filter to just the most relevant columns for a cleaner correlation matrix
        relevant_cols = ['property_value', 'property_size', 'property_age', 
                         'flood_risk_score', 'fire_risk_score', 'location_risk']
        relevant_cols = [col for col in relevant_cols if col in numeric_cols]
        corr = property_data[relevant_cols].corr()
        
        # Rename columns for better readability
        readable_names = {
            'property_value': 'Property Value',
            'property_size': 'Property Size',
            'property_age': 'Property Age',
            'flood_risk_score': 'Flood Risk',
            'fire_risk_score': 'Fire Risk',
            'location_risk': 'Location Risk'
        }
        
        corr_renamed = corr.rename(index=readable_names, columns=readable_names)
        
        # Create a human-friendly heatmap with clear text
        heatmap_values = corr_renamed.values
        
        # Use a lighter color scale to ensure text visibility
        custom_colorscale = [
            [0.0, '#4575b4'],  # Dark blue for strong negative
            [0.3, '#abd9e9'],  # Light blue for weak negative
            [0.5, '#f7f7f7'],  # White for neutral
            [0.7, '#fdae61'],  # Light orange for weak positive
            [1.0, '#d73027']   # Dark red for strong positive
        ]
        
        # Create figure with a simple heatmap
        fig = go.Figure()
        
        # Add heatmap trace with improved configuration
        fig.add_trace(go.Heatmap(
            z=heatmap_values,
            x=list(corr_renamed.columns),
            y=list(corr_renamed.index),
            colorscale=custom_colorscale,
            zmid=0,  # Center the color scale at 0
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title='Correlation',
                titleside='right',
                ticks='outside',
                tickfont=dict(size=12),
                titlefont=dict(size=14)
            )
        ))
        
        # Add text annotations with improved visibility
        for i in range(len(corr_renamed.index)):
            for j in range(len(corr_renamed.columns)):
                value = heatmap_values[i, j]
                # Determine text color based on background intensity
                # Use white for dark backgrounds, black for light backgrounds
                text_color = "white" if abs(value) > 0.4 else "black"
                
                # Add bold outline to text for better visibility
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color=text_color,
                        size=14,
                        family="Arial Black"
                    ),
                    bgcolor="rgba(255,255,255,0.0)",  # Transparent background
                    borderpad=2
                )
        
        # Update layout with improved settings for human readability
        fig.update_layout(
            title={
                'text': "Risk Factor Correlation Matrix",
                'font': {'size': 18, 'family': 'Arial', 'color': '#333'}
            },
            height=500,  # Fixed height
            margin=dict(l=50, r=50, t=70, b=50),  # Better margins
            xaxis=dict(
                title='',
                tickangle=-30,  # Angle the labels for better readability
                tickfont=dict(size=13)
            ),
            yaxis=dict(
                title='',
                tickfont=dict(size=13)
            ),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Display the chart with container width
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_correlation)

def render_risk_relationships(property_data):
    """Render visualizations showing key risk relationships."""
    st.markdown('<h4 class="section-header">Key Risk Relationships</h4>', unsafe_allow_html=True)
    
    def plot_risk_relationships():
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter to properties with valid risk scores
            filtered_property_data = property_data.copy()
            
            # Box plot of risk scores by property type
            fig = px.box(
                filtered_property_data, 
                x='property_type', 
                y='location_risk',
                color='property_type',
                color_discrete_map=get_property_type_colors(),
                points="all",  # Show all points
                labels={
                    'property_type': 'Property Type',
                    'location_risk': 'Location Risk Score'
                },
                title="Location Risk by Property Type"
            )
            
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of property value vs location risk
            fig = px.scatter(
                property_data,
                x='property_value',
                y='location_risk',
                color='property_type',
                title='Property Value vs Location Risk',
                color_discrete_map=get_property_type_colors(),
                opacity=0.7,
                # Removed trendline='ols' to avoid statsmodels dependency
                labels={
                    'property_value': 'Property Value ($)',
                    'location_risk': 'Location Risk Score',
                    'property_type': 'Property Type'
                }
            )
            
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_risk_relationships)

def render_risk_analysis_page(property_data):
    """Render the complete Risk Analysis page."""
    st.markdown('<h2 class="section-header">Risk Analysis</h2>', unsafe_allow_html=True)
    
    # Render the correlation heatmap
    render_correlation_heatmap(property_data)
    
    # Render the risk relationships
    render_risk_relationships(property_data)
