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
        
        # Create heatmap with simplified configuration
        fig = px.imshow(
            corr_renamed,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect="auto"
        )
        
        # Basic layout updates that are known to work
        fig.update_layout(
            height=500,
            title="Risk Factor Correlation Matrix",
            font=dict(color='#0F172A'),
            plot_bgcolor='white'
        )
        
        # Simplify colorbar configuration to avoid nested structure issues
        fig.update_coloraxes(colorbar=dict(
            thickness=20,
            len=400,
            y=1,
            yanchor="top",
            ticks="outside",
            tickfont=dict(color='#0F172A')
        ))
        
        # Ensure text is visible and properly formatted
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textfont=dict(size=12, color='black')
        )
        
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
                trendline='ols',
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
